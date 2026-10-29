//! Thin compatibility layer that re-exports either `rayon` or a sequential
//! fallback under a unified prelude, gated by the `parallel` feature.

#![no_std]

extern crate alloc;

#[cfg(any(feature = "parallel", test))]
extern crate std;

/// Whether this crate uses the Rayon backend instead of the serial fallback.
///
/// This reflects feature unification across dependents and remains true for a
/// one-worker Rayon pool, unlike a check of the current worker count.
pub const PARALLEL_ENABLED: bool = cfg!(feature = "parallel");

#[cfg(not(feature = "parallel"))]
mod serial;

#[cfg(feature = "parallel")]
mod dispatch;

/// A small dedicated thread pool for latency-sensitive parallel sections.
///
/// A parallel search launched into the global pool queues behind whatever long-running
/// parallel sections the process already has in flight, so its latency scales with process
/// load rather than with its own work. `dedicated_install` runs the closure on a lazily-built
/// private pool when `parallel` is on, and inline when it is off.
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

pub mod search;
pub mod task_size;

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
    pub use super::task_size::{TaskSizeExt, min_task_len, should_split};

    pub trait SharedExt: ParallelIterator {
        /// Folds each split of the iterator with `fold_op` (seeded by `identity`), then
        /// combines the per-split accumulators with `reduce_op`.
        ///
        /// `identity()` must be neutral for `reduce_op`, and `fold_op`/`reduce_op` must be
        /// mutually associative so that regrouping splits doesn't change the result — under
        /// the `parallel` feature the split count (and thus how many times `reduce_op` runs)
        /// depends on the thread pool. **The serial fallback never calls `reduce_op`** (there
        /// is only one split), so an inconsistent `reduce_op` yields results that diverge
        /// between `cargo test` and `cargo test --features parallel`.
        fn par_fold_reduce<Acc, Id, F, R>(self, identity: Id, fold_op: F, reduce_op: R) -> Acc
        where
            Acc: Send,
            Id: Fn() -> Acc + Sync + Send,
            F: Fn(Acc, Self::Item) -> Acc + Sync + Send,
            R: Fn(Acc, Acc) -> Acc + Sync + Send;
    }

    #[cfg(feature = "parallel")]
    impl<I: IndexedParallelIterator> SharedExt for I {
        #[inline]
        fn par_fold_reduce<Acc, Id, F, R>(self, identity: Id, fold_op: F, reduce_op: R) -> Acc
        where
            Acc: Send,
            Id: Fn() -> Acc + Sync + Send,
            F: Fn(Acc, Self::Item) -> Acc + Sync + Send,
            R: Fn(Acc, Acc) -> Acc + Sync + Send,
        {
            super::dispatch::fold_reduce(self, identity, fold_op, reduce_op)
        }
    }

    #[cfg(not(feature = "parallel"))]
    impl<I: ParallelIterator> SharedExt for I {
        #[inline]
        fn par_fold_reduce<Acc, Id, F, R>(self, identity: Id, fold_op: F, reduce_op: R) -> Acc
        where
            Acc: Send,
            Id: Fn() -> Acc + Sync + Send,
            F: Fn(Acc, Self::Item) -> Acc + Sync + Send,
            R: Fn(Acc, Acc) -> Acc + Sync + Send,
        {
            let _ = reduce_op;
            self.fold(identity(), fold_op)
        }
    }
}

pub mod iter {
    #[cfg(not(feature = "parallel"))]
    pub use core::iter::{repeat, repeat_n};

    #[cfg(feature = "parallel")]
    pub use rayon::iter::{repeat, repeat_n};
}
