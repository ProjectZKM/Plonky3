#![no_std]

#[cfg(feature = "parallel")]
extern crate std;

#[cfg(not(feature = "parallel"))]
mod serial;

/// A small DEDICATED thread pool for latency-sensitive parallel sections.
///
/// A parallel search launched into the GLOBAL pool queues behind whatever
/// long-running parallel sections the process already has in flight, so its
/// latency scales with process load rather than with its own work (measured:
/// a proof-of-work grind whose compute is sub-millisecond spending ~40 ms
/// waiting for pool slots).  `dedicated_install` runs the closure on a
/// lazily-built private pool when `parallel` is on, and inline when it is
/// off.
pub mod pool {
    #[cfg(feature = "parallel")]
    pub fn dedicated_install<R: Send>(threads: usize, f: impl FnOnce() -> R + Send) -> R {
        use std::sync::OnceLock;
        static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();
        POOL.get_or_init(|| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .thread_name(|i| std::format!("p3-dedicated-{i}"))
                .build()
                .expect("dedicated pool")
        })
        .install(f)
    }

    #[cfg(not(feature = "parallel"))]
    pub fn dedicated_install<R>(threads: usize, f: impl FnOnce() -> R) -> R {
        let _ = threads;
        f()
    }
}

pub mod prelude {
    #[cfg(not(feature = "parallel"))]
    pub use core::iter::{
        ExactSizeIterator as IndexedParallelIterator, Iterator as ParallelIterator,
    };
    use core::marker::{Send, Sync};

    #[cfg(feature = "parallel")]
    pub use rayon::prelude::*;
    #[cfg(feature = "parallel")]
    pub use rayon::{current_num_threads, join};

    #[cfg(not(feature = "parallel"))]
    pub use super::serial::*;

    pub trait SharedExt: ParallelIterator {
        fn par_fold_reduce<Acc, Id, F, R>(self, identity: Id, fold_op: F, reduce_op: R) -> Acc
        where
            Acc: Send,
            Id: Fn() -> Acc + Sync + Send,
            F: Fn(Acc, Self::Item) -> Acc + Sync + Send,
            R: Fn(Acc, Acc) -> Acc + Sync + Send;
    }

    impl<I: ParallelIterator> SharedExt for I {
        #[inline]
        fn par_fold_reduce<Acc, Id, F, R>(self, identity: Id, fold_op: F, reduce_op: R) -> Acc
        where
            Acc: Send,
            Id: Fn() -> Acc + Sync + Send,
            F: Fn(Acc, Self::Item) -> Acc + Sync + Send,
            R: Fn(Acc, Acc) -> Acc + Sync + Send,
        {
            #[cfg(feature = "parallel")]
            {
                self.fold(&identity, fold_op).reduce(&identity, reduce_op)
            }

            #[cfg(not(feature = "parallel"))]
            {
                let _ = reduce_op;
                self.fold(identity(), fold_op)
            }
        }
    }
}

pub mod iter {
    #[cfg(not(feature = "parallel"))]
    pub use core::iter::{repeat, repeat_n};

    #[cfg(feature = "parallel")]
    pub use rayon::iter::{repeat, repeat_n};
}

/// A raw mutable pointer wrapper that implements [`Send`] and [`Sync`].
///
/// Used to enable parallel writes to disjoint slices of a pre-allocated buffer
/// from within closures that require `Send + Sync` (e.g. [`rayon::ParallelIterator::for_each_init`]).
///
/// # Safety
///
/// The caller must ensure that concurrent accesses through this pointer always
/// target **non-overlapping** memory regions.
#[derive(Clone, Copy)]
pub struct DisjointMutPtr<T>(*mut T);

// SAFETY: The contract of DisjointMutPtr guarantees that each thread writes to
// a disjoint region, so sharing the pointer across threads is safe.
unsafe impl<T> Send for DisjointMutPtr<T> {}
unsafe impl<T> Sync for DisjointMutPtr<T> {}

impl<T> DisjointMutPtr<T> {
    /// Create a new `DisjointMutPtr` from a mutable slice.
    #[inline]
    pub const fn new(slice: &mut [T]) -> Self {
        Self(slice.as_mut_ptr())
    }

    /// Get a mutable slice starting at `offset` with `len` elements.
    ///
    /// # Safety
    ///
    /// The caller must ensure the range `[offset, offset+len)` is within bounds
    /// and does not overlap with any other concurrent access.
    #[inline]
    pub const unsafe fn slice_mut(self, offset: usize, len: usize) -> &'static mut [T] {
        unsafe { core::slice::from_raw_parts_mut(self.0.add(offset), len) }
    }
}
