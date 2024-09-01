//! An allocator that can track and limit memory usage.
//!
//! **[Crates.io](https://crates.io/crates/cap) │ [Repo](https://github.com/alecmocatta/cap)**
//!
//! This crate provides a generic allocator that wraps another allocator, tracking memory usage and enabling limits to be set.
//!
//! # Example
//!
//! It can be used by declaring a static and marking it with the `#[global_allocator]` attribute:
//!
//! ```
//! use std::alloc;
//! use cap::Cap;
//!
//! #[global_allocator]
//! static ALLOCATOR: Cap<alloc::System> = Cap::new(alloc::System, usize::max_value());
//!
//! fn main() {
//!     // Set the limit to 30MiB.
//!     ALLOCATOR.set_limit(30 * 1024 * 1024).unwrap();
//!     // ...
//!     println!("Currently allocated: {}B", ALLOCATOR.allocated());
//! }
//! ```

#![cfg_attr(feature = "nightly", feature(allocator_api))]
#![cfg_attr(
	all(test, feature = "nightly"),
	feature(try_reserve, test, custom_test_frameworks)
)]
#![cfg_attr(all(test, feature = "nightly"), test_runner(tests::runner))]
#![warn(
	missing_copy_implementations,
	missing_debug_implementations,
	missing_docs,
	trivial_casts,
	trivial_numeric_casts,
	unused_import_braces,
	unused_qualifications,
	unused_results,
	clippy::pedantic
)] // from https://github.com/rust-unofficial/patterns/blob/master/anti_patterns/deny-warnings.md
#![allow(
	clippy::result_unit_err,
	clippy::let_underscore_untyped,
	clippy::missing_errors_doc
)]

#[cfg(feature = "nightly")]
use std::alloc::{AllocError, Allocator};
#[cfg(all(not(feature = "nightly"), feature = "allocator-api2"))]
use allocator_api2::alloc::{AllocError, Allocator};
use std::{
	alloc::{GlobalAlloc, Layout}, ptr, sync::atomic::{AtomicUsize, Ordering}
};

/// A struct that wraps another allocator and limits the number of bytes that can be allocated.
#[derive(Debug)]
pub struct Cap<H> {
	allocator: H,
	remaining: AtomicUsize,
	limit: AtomicUsize,
	#[cfg(feature = "stats")]
	total_allocated: AtomicUsize,
	#[cfg(feature = "stats")]
	max_allocated: AtomicUsize,
}

impl<H> Cap<H> {
	/// Create a new allocator, wrapping the supplied allocator and enforcing the specified limit.
	///
	/// For no limit, simply set the limit to the theoretical maximum `usize::max_value()`.
	pub const fn new(allocator: H, limit: usize) -> Self {
		Self {
			allocator,
			remaining: AtomicUsize::new(limit),
			limit: AtomicUsize::new(limit),
			#[cfg(feature = "stats")]
			total_allocated: AtomicUsize::new(0),
			#[cfg(feature = "stats")]
			max_allocated: AtomicUsize::new(0),
		}
	}

	/// Return the number of bytes remaining within the limit.
	///
	/// i.e. `limit - allocated`
	pub fn remaining(&self) -> usize {
		self.remaining.load(Ordering::Relaxed)
	}

	/// Return the limit in bytes.
	pub fn limit(&self) -> usize {
		self.limit.load(Ordering::Relaxed)
	}

	/// Set the limit in bytes.
	///
	/// For no limit, simply set the limit to the theoretical maximum `usize::max_value()`.
	///
	/// This method will return `Err` if the specified limit is less than the number of bytes already allocated.
	pub fn set_limit(&self, limit: usize) -> Result<(), ()> {
		loop {
			let limit_old = self.limit.load(Ordering::Relaxed);
			if limit < limit_old {
				if self
					.remaining
					.fetch_sub(limit_old - limit, Ordering::Relaxed)
					< limit_old - limit
				{
					let _ = self
						.remaining
						.fetch_add(limit_old - limit, Ordering::Relaxed);
					break Err(());
				}
				if self
					.limit
					.compare_exchange(limit_old, limit, Ordering::Relaxed, Ordering::Relaxed)
					.is_err()
				{
					continue;
				}
			} else {
				if self
					.limit
					.compare_exchange(limit_old, limit, Ordering::Relaxed, Ordering::Relaxed)
					.is_err()
				{
					continue;
				}
				let _ = self
					.remaining
					.fetch_add(limit - limit_old, Ordering::Relaxed);
			}
			break Ok(());
		}
	}

	/// Return the number of bytes allocated. Always less than the limit.
	pub fn allocated(&self) -> usize {
		// Make reasonable effort to get valid output
		loop {
			let limit_old = self.limit.load(Ordering::SeqCst);
			let remaining = self.remaining.load(Ordering::SeqCst);
			let limit = self.limit.load(Ordering::SeqCst);
			if limit_old == limit && limit >= remaining {
				break limit - remaining;
			}
		}
	}

	/// Get total amount of allocated memory. This includes already deallocated memory.
	#[cfg(feature = "stats")]
	pub fn total_allocated(&self) -> usize {
		self.total_allocated.load(Ordering::Relaxed)
	}

	/// Get maximum amount of memory that was allocated at any point in time.
	#[cfg(feature = "stats")]
	pub fn max_allocated(&self) -> usize {
		self.max_allocated.load(Ordering::Relaxed)
	}

	fn update_stats(&self, size: usize) {
		#[cfg(feature = "stats")]
		{
			let _ = self.total_allocated.fetch_add(size, Ordering::Relaxed);
			// If max_allocated is less than currently allocated, then it will be updated to limit - remaining.
			// Otherwise, it will remain unchanged.
			let _ = self
				.max_allocated
				.fetch_max(self.allocated(), Ordering::Relaxed);
		}
		#[cfg(not(feature = "stats"))]
		{
			let _ = (self, size);
		}
	}
}

unsafe impl<H> GlobalAlloc for Cap<H>
where
	H: GlobalAlloc,
{
	unsafe fn alloc(&self, l: Layout) -> *mut u8 {
		let size = l.size();
		let res = if self.remaining.fetch_sub(size, Ordering::Acquire) >= size {
			self.allocator.alloc(l)
		} else {
			ptr::null_mut()
		};
		if res.is_null() {
			let _ = self.remaining.fetch_add(size, Ordering::Release);
		} else {
			self.update_stats(size);
		}
		res
	}
	unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
		let size = layout.size();
		self.allocator.dealloc(ptr, layout);
		let _ = self.remaining.fetch_add(size, Ordering::Release);
	}
	unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
		let size = l.size();
		let res = if self.remaining.fetch_sub(size, Ordering::Acquire) >= size {
			self.allocator.alloc_zeroed(l)
		} else {
			ptr::null_mut()
		};
		if res.is_null() {
			let _ = self.remaining.fetch_add(size, Ordering::Release);
		} else {
			self.update_stats(size);
		}
		res
	}
	unsafe fn realloc(&self, ptr: *mut u8, old_l: Layout, new_s: usize) -> *mut u8 {
		let new_l = Layout::from_size_align_unchecked(new_s, old_l.align());
		let (old_size, new_size) = (old_l.size(), new_l.size());
		let res = if new_size > old_size {
			let res = if self
				.remaining
				.fetch_sub(new_size - old_size, Ordering::Acquire)
				>= new_size - old_size
			{
				self.allocator.realloc(ptr, old_l, new_s)
			} else {
				ptr::null_mut()
			};
			if res.is_null() {
				let _ = self
					.remaining
					.fetch_add(new_size - old_size, Ordering::Release);
			}
			res
		} else {
			let res = self.allocator.realloc(ptr, old_l, new_s);
			if !res.is_null() {
				let _ = self
					.remaining
					.fetch_add(old_size - new_size, Ordering::Release);
			}
			// Although this might just deaalocate, I will still update the stats as if it allocates to be on "the safe side"
			res
		};
		if !res.is_null() {
			self.update_stats(new_size);
		}
		res
	}
}

#[cfg(any(feature = "nightly", feature = "allocator-api2"))]
unsafe impl<H> Allocator for Cap<H>
where
	H: Allocator,
{
	fn allocate(&self, l: Layout) -> Result<ptr::NonNull<[u8]>, AllocError> {
		let size = l.size();
		let res = if self.remaining.fetch_sub(size, Ordering::Acquire) >= size {
			self.allocator.allocate(l)
		} else {
			Err(AllocError)
		};
		if res.is_err() {
			let _ = self.remaining.fetch_add(size, Ordering::Release);
		} else {
			self.update_stats(size);
		}
		res
	}

	fn allocate_zeroed(&self, l: Layout) -> Result<ptr::NonNull<[u8]>, AllocError> {
		let size = l.size();
		let res = if self.remaining.fetch_sub(size, Ordering::Acquire) >= size {
			self.allocator.allocate_zeroed(l)
		} else {
			Err(AllocError)
		};
		if res.is_err() {
			let _ = self.remaining.fetch_add(size, Ordering::Release);
		} else {
			self.update_stats(size);
		}
		res
	}

	unsafe fn deallocate(&self, item: ptr::NonNull<u8>, l: Layout) {
		let size = l.size();
		self.allocator.deallocate(item, l);
		let _ = self.remaining.fetch_add(size, Ordering::Release);
	}

	unsafe fn grow(
		&self, ptr: ptr::NonNull<u8>, old_l: Layout, new_l: Layout,
	) -> Result<ptr::NonNull<[u8]>, AllocError> {
		let additional = new_l.size() - old_l.size();
		let res = if self
			.remaining
			.fetch_sub(additional, Ordering::Acquire)
			>= additional
		{
			self.allocator.grow(ptr, old_l, new_l)
		} else {
			Err(AllocError)
		};
		if res.is_err() {
			let _ = self
				.remaining
				.fetch_add(additional, Ordering::Release);
		} else {
			self.update_stats(additional);
		}
		res
	}

	unsafe fn grow_zeroed(&self, ptr: ptr::NonNull<u8>, old_l: Layout, new_l: Layout) -> Result<ptr::NonNull<[u8]>, AllocError> {
		let (old_size, new_size) = (
			old_l.size(),
			new_l.size(),
		);
		let res = if self
			.remaining
			.fetch_sub(new_size - old_size, Ordering::Acquire)
			>= new_size - old_size
		{
			self.allocator.grow_zeroed(ptr, old_l, new_l)
		} else {
			Err(AllocError)
		};
		if res.is_err() {
			let _ = self
				.remaining
				.fetch_add(new_size - old_size, Ordering::Release);
		} else {
			self.update_stats(new_size - old_size);
		}
		res
	}

	unsafe fn shrink(
		&self, ptr: ptr::NonNull<u8>, old_l: Layout, new_l: Layout,
	) -> Result<ptr::NonNull<[u8]>, AllocError> {
		let removed = old_l.size() - new_l.size();
		let res = self.allocator.shrink(ptr, old_l, new_l);
		if res.is_ok() {
			let _ = self
				.remaining
				.fetch_add(removed, Ordering::Release);
		}
		res
	}
}

#[cfg(test)]
mod tests {
	#[cfg_attr(all(test, feature = "nightly"), feature(allocator_api))]
	#[cfg(all(test, feature = "nightly"))]
	extern crate test;
	#[cfg(all(test))]
	use std::collections::TryReserveError;
	use std::{alloc, thread};
	#[cfg(all(test, feature = "nightly"))]
	use test::{TestDescAndFn, TestFn};
	#[cfg(all(test, not(feature = "nightly"), feature = "allocator-api2"))]
	use allocator_api2::vec::Vec;

	use super::Cap;

	#[global_allocator]
	static A: Cap<alloc::System> = Cap::new(alloc::System, usize::max_value());

	#[cfg(all(test, feature = "nightly"))]
	pub fn runner(tests: &[&TestDescAndFn]) {
		for test in tests {
			if let TestFn::StaticTestFn(test_fn) = test.testfn {
				test_fn();
			} else {
				unimplemented!();
			}
		}
	}

	#[test]
	fn concurrent() {
		let allocated = A.allocated();
		for _ in 0..100 {
			let threads = (0..100)
				.map(|_| {
					thread::spawn(|| {
						for i in 0..1000 {
							let _ = (0..i).collect::<Vec<u32>>();
							let _ = (0..i).flat_map(std::iter::once).collect::<Vec<u32>>();
						}
					})
				})
				.collect::<Vec<_>>();
			threads
				.into_iter()
				.for_each(|thread| thread.join().unwrap());
			let allocated2 = A.allocated();
			#[cfg(feature = "stats")]
			let total_allocated = A.total_allocated();
			if cfg!(all(test, feature = "nightly")) {
				assert_eq!(allocated, allocated2);
				#[cfg(feature = "stats")]
				assert!(total_allocated >= allocated);
			}
		}
		#[cfg(feature = "stats")]
		assert!(A.max_allocated() < A.total_allocated());
	}

	#[cfg(all(test, not(any(feature = "nightly", feature = "allocator-api2"))))]
	#[test]
	fn limit() {
		#[cfg(feature = "stats")]
		let initial = A.allocated();
		let allocate_amount = 30 * 1024 * 1024;
		A.set_limit(A.allocated() + allocate_amount).unwrap();
		for _ in 0..10 {
			let mut vec = Vec::<u8>::with_capacity(0);
			if let Err(_e) = vec.try_reserve_exact(allocate_amount + 1) {} else {
				A.set_limit(usize::max_value()).unwrap();
				panic!("{}", A.remaining());
			};
			assert_eq!(vec.try_reserve_exact(allocate_amount), Ok(()));
			let mut vec2 = Vec::<u8>::with_capacity(0);
			assert!(vec2.try_reserve_exact(1).is_err());
		}
		// Might have additional allocations of errors and what not along the way.
		#[cfg(feature = "stats")]
		{
			assert!(A.total_allocated() >= initial + 10 * allocate_amount);
			assert_eq!(A.max_allocated(), initial + allocate_amount);
		}
	}

	#[cfg(all(test, any(feature = "nightly", feature = "allocator-api2")))]
	#[test]
	fn limit() {
		let allocate_amount = 30 * 1024 * 1024;
		let allocated = A.allocated();
		A.set_limit(allocated + allocate_amount).unwrap();
		for _ in 0..10 {
			let mut vec = Vec::<u8, _>::with_capacity_in(0, &A);
			if vec.try_reserve_exact(allocate_amount + 1).is_err() {
			} else {
				A.set_limit(usize::max_value()).unwrap();
				panic!("{}", A.remaining())
			};
			assert_eq!(vec.try_reserve_exact(allocate_amount), Ok(()));
			let mut vec2 = Vec::<u8, _>::with_capacity_in(0, &A);
			assert!(vec2.try_reserve_exact(1).is_err());
		}
		assert_eq!(A.max_allocated(), allocate_amount + allocated);
		assert!(A.total_allocated() >= allocated + (10 * allocate_amount));
	}
}
