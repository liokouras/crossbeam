use std::boxed::Box;
use std::cell::Cell;
use std::fmt;
use std::marker::PhantomData;
use std::mem::{self, MaybeUninit};
use std::ptr;
use std::sync::atomic::{self, AtomicIsize, Ordering};
use std::sync::Arc;

use crossbeam_epoch::{self as epoch, Atomic, Owned};
use crossbeam_utils::CachePadded;

// Minimum buffer capacity.
const MIN_CAP: usize = 64;

// If a buffer of at least this size is retired, thread-local garbage is flushed so that it gets
// deallocated as soon as possible.
const FLUSH_THRESHOLD_BYTES: usize = 1 << 10;

/// A buffer that holds tasks in a worker queue.
///
/// This is just a pointer to the buffer and its length - dropping an instance of this struct will
/// *not* deallocate the buffer.
struct Buffer<T> {
    /// Pointer to the allocated memory.
    ptr: *mut T,

    /// Capacity of the buffer. Always a power of two.
    cap: usize,
}

unsafe impl<T> Send for Buffer<T> {}

impl<T> Buffer<T> {
    /// Allocates a new buffer with the specified capacity.
    fn alloc(cap: usize) -> Self {
        debug_assert_eq!(cap, cap.next_power_of_two());

        let ptr = Box::into_raw(
            (0..cap)
                .map(|_| MaybeUninit::<T>::uninit())
                .collect::<Box<[_]>>(),
        )
        .cast::<T>();

        Self { ptr, cap }
    }

    /// Deallocates the buffer.
    unsafe fn dealloc(self) {
        drop(unsafe {
            Box::from_raw(ptr::slice_from_raw_parts_mut(
                self.ptr.cast::<MaybeUninit<T>>(),
                self.cap,
            ))
        });
    }

    /// Returns a pointer to the task at the specified `index`.
    unsafe fn at(&self, index: isize) -> *mut T {
        // `self.cap` is always a power of two.
        // We do all the loads at `MaybeUninit` because we might realize, after loading, that we
        // don't actually have the right to access this memory.
        unsafe { self.ptr.offset(index & (self.cap - 1) as isize) }
    }

    /// Writes `task` into the specified `index`.
    ///
    /// This method might be concurrently called with another `read` at the same index, which is
    /// technically speaking a data race and therefore UB. We should use an atomic store here, but
    /// that would be more expensive and difficult to implement generically for all types `T`.
    /// Hence, as a hack, we use a volatile write instead.
    unsafe fn write(&self, index: isize, task: MaybeUninit<T>) {
        unsafe { ptr::write_volatile(self.at(index).cast::<MaybeUninit<T>>(), task) }
    }

    /// Reads a task from the specified `index`.
    ///
    /// This method might be concurrently called with another `write` at the same index, which is
    /// technically speaking a data race and therefore UB. We should use an atomic load here, but
    /// that would be more expensive and difficult to implement generically for all types `T`.
    /// Hence, as a hack, we use a volatile load instead.
    unsafe fn read(&self, index: isize) -> MaybeUninit<T> {
        unsafe { ptr::read_volatile(self.at(index).cast::<MaybeUninit<T>>()) }
    }
}

impl<T> Clone for Buffer<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Buffer<T> {}

/// Internal queue data shared between the worker and stealers.
///
/// The implementation is based on the following work:
///
/// 1. [Chase and Lev. Dynamic circular work-stealing deque. SPAA 2005.][chase-lev]
/// 2. [Le, Pop, Cohen, and Nardelli. Correct and efficient work-stealing for weak memory models.
///    PPoPP 2013.][weak-mem]
/// 3. [Norris and Demsky. CDSchecker: checking concurrent data structures written with C/C++
///    atomics. OOPSLA 2013.][checker]
///
/// [chase-lev]: https://dl.acm.org/citation.cfm?id=1073974
/// [weak-mem]: https://dl.acm.org/citation.cfm?id=2442524
/// [checker]: https://dl.acm.org/citation.cfm?id=2509514
struct Inner<T> {
    /// The front index.
    front: AtomicIsize,

    /// The back index.
    back: AtomicIsize,

    /// The underlying buffer.
    buffer: CachePadded<Atomic<Buffer<T>>>,
}

impl<T> Drop for Inner<T> {
    fn drop(&mut self) {
        // Load the back index, front index, and buffer.
        let b = *self.back.get_mut();
        let f = *self.front.get_mut();

        unsafe {
            let buffer = self.buffer.load(Ordering::Relaxed, epoch::unprotected());

            // Go through the buffer from front to back and drop all tasks in the queue.
            let mut i = f;
            while i != b {
                buffer.deref().at(i).drop_in_place();
                i = i.wrapping_add(1);
            }

            // Free the memory allocated by the buffer.
            buffer.into_owned().into_box().dealloc();
        }
    }
}

/// A worker queue.
///
/// This is a FIFO or LIFO queue that is owned by a single thread, but other threads may steal
/// tasks from it. Task schedulers typically create a single worker queue per thread.
///
/// # Examples
///
/// A FIFO worker:
///
/// ```
/// use crossbeam_deque::{Steal, Worker};
///
/// let w = Worker::new_fifo();
/// let s = w.stealer();
///
/// w.push(1);
/// w.push(2);
/// w.push(3);
///
/// assert_eq!(s.steal(), Steal::Success(1));
/// assert_eq!(w.pop(), Some(2));
/// assert_eq!(w.pop(), Some(3));
/// ```
///
/// A LIFO worker:
///
/// ```
/// use crossbeam_deque::{Steal, Worker};
///
/// let w = Worker::new_lifo();
/// let s = w.stealer();
///
/// w.push(1);
/// w.push(2);
/// w.push(3);
///
/// assert_eq!(s.steal(), Steal::Success(1));
/// assert_eq!(w.pop(), Some(3));
/// assert_eq!(w.pop(), Some(2));
/// ```
pub struct Worker<T> {
    /// A reference to the inner representation of the queue.
    inner: Arc<CachePadded<Inner<T>>>,

    /// A copy of `inner.buffer` for quick access.
    buffer: Cell<Buffer<T>>,

    /// Indicates that the worker cannot be shared among threads.
    _marker: PhantomData<*mut ()>, // !Send + !Sync
}

unsafe impl<T: Send> Send for Worker<T> {}

impl<T> Worker<T> {
    /// Creates a LIFO worker queue.
    ///
    /// Tasks are pushed and popped from the same end.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbeam_deque::Worker;
    ///
    /// let w = Worker::<i32>::new_lifo();
    /// ```
    pub fn new_lifo() -> Self {
        let buffer = Buffer::alloc(MIN_CAP);

        let inner = Arc::new(CachePadded::new(Inner {
            front: AtomicIsize::new(0),
            back: AtomicIsize::new(0),
            buffer: CachePadded::new(Atomic::new(buffer)),
        }));

        Self {
            inner,
            buffer: Cell::new(buffer),
            _marker: PhantomData,
        }
    }

    /// Creates a stealer for this queue.
    ///
    /// The returned stealer can be shared among threads and cloned.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbeam_deque::Worker;
    ///
    /// let w = Worker::<i32>::new_lifo();
    /// let s = w.stealer();
    /// ```
    pub fn stealer(&self) -> Stealer<T> {
        Stealer {
            inner: self.inner.clone(),
        }
    }

    /// Resizes the internal buffer to the new capacity of `new_cap`.
    #[cold]
    unsafe fn resize(&self, new_cap: usize) {
        // Load the back index, front index, and buffer.
        let b = self.inner.back.load(Ordering::Relaxed);
        let f = self.inner.front.load(Ordering::Relaxed);
        let buffer = self.buffer.get();

        // Allocate a new buffer and copy data from the old buffer to the new one.
        let new = Buffer::alloc(new_cap);
        let mut i = f;
        while i != b {
            unsafe { ptr::copy_nonoverlapping(buffer.at(i), new.at(i), 1) }
            i = i.wrapping_add(1);
        }

        let guard = &epoch::pin();

        // Replace the old buffer with the new one.
        self.buffer.replace(new);
        let old =
            self.inner
                .buffer
                .swap(Owned::new(new).into_shared(guard), Ordering::Release, guard);

        // Destroy the old buffer later.
        unsafe { guard.defer_unchecked(move || old.into_owned().into_box().dealloc()) }

        // If the buffer is very large, then flush the thread-local garbage in order to deallocate
        // it as soon as possible.
        if mem::size_of::<T>() * new_cap >= FLUSH_THRESHOLD_BYTES {
            guard.flush();
        }
    }

    /// Returns `true` if the queue is empty.
    ///
    /// ```
    /// use crossbeam_deque::Worker;
    ///
    /// let w = Worker::new_lifo();
    ///
    /// assert!(w.is_empty());
    /// w.push(1);
    /// assert!(!w.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        let b = self.inner.back.load(Ordering::Relaxed);
        let f = self.inner.front.load(Ordering::SeqCst);
        b.wrapping_sub(f) <= 0
    }

    /// Returns the number of tasks in the deque.
    ///
    /// ```
    /// use crossbeam_deque::Worker;
    ///
    /// let w = Worker::new_lifo();
    ///
    /// assert_eq!(w.len(), 0);
    /// w.push(1);
    /// assert_eq!(w.len(), 1);
    /// w.push(1);
    /// assert_eq!(w.len(), 2);
    /// ```
    pub fn len(&self) -> usize {
        let b = self.inner.back.load(Ordering::Relaxed);
        let f = self.inner.front.load(Ordering::SeqCst);
        b.wrapping_sub(f).max(0) as usize
    }

    /// Pushes a task into the queue.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbeam_deque::Worker;
    ///
    /// let w = Worker::new_lifo();
    /// w.push(1);
    /// w.push(2);
    /// ```
    pub fn push(&self, task: T) {
        // Load the back index, front index, and buffer.
        let b = self.inner.back.load(Ordering::Relaxed);
        let f = self.inner.front.load(Ordering::Acquire);
        let mut buffer = self.buffer.get();

        // Calculate the length of the queue.
        let len = b.wrapping_sub(f);

        // Is the queue full?
        if len >= buffer.cap as isize {
            // Yes. Grow the underlying buffer.
            unsafe {
                self.resize(2 * buffer.cap);
            }
            buffer = self.buffer.get();
        }

        // Write `task` into the slot.
        unsafe {
            buffer.write(b, MaybeUninit::new(task));
        }

        atomic::fence(Ordering::Release);

        // Increment the back index.
        //
        // This ordering could be `Relaxed`, but then thread sanitizer would falsely report data
        // races because it doesn't understand fences.
        self.inner.back.store(b.wrapping_add(1), Ordering::Release);
    }

    /// Pops a task from the queue.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbeam_deque::Worker;
    ///
    /// let w = Worker::new_fifo();
    /// w.push(1);
    /// w.push(2);
    ///
    /// assert_eq!(w.pop(), Some(1));
    /// assert_eq!(w.pop(), Some(2));
    /// assert_eq!(w.pop(), None);
    /// ```
    pub fn pop(&self) -> Option<T> {
        // Load the back and front index.
        let b = self.inner.back.load(Ordering::Relaxed);
        let f = self.inner.front.load(Ordering::Relaxed);

        // Calculate the length of the queue.
        let len = b.wrapping_sub(f);

        // Is the queue empty?
        if len <= 0 {
            return None;
        }

        // Decrement the back index.
        let b = b.wrapping_sub(1);
        self.inner.back.store(b, Ordering::Relaxed);

        atomic::fence(Ordering::SeqCst);

        // Load the front index.
        let f = self.inner.front.load(Ordering::Relaxed);

        // Compute the length after the back index was decremented.
        let len = b.wrapping_sub(f);

        if len < 0 {
            // The queue is empty. Restore the back index to the original task.
            self.inner.back.store(b.wrapping_add(1), Ordering::Relaxed);
            None
        } else {
            // Read the task to be popped.
            let buffer = self.buffer.get();
            let mut task = unsafe { Some(buffer.read(b)) };

            // Are we popping the last task from the queue?
            // BAZI TODO: have we overtaken 'stealable'?
            if len == 0 {
                // Try incrementing the front index.
                if self
                    .inner
                    .front
                    .compare_exchange(
                        f,
                        f.wrapping_add(1),
                        Ordering::SeqCst,
                        Ordering::Relaxed,
                    )
                    .is_err()
                {
                    // Failed. We didn't pop anything. Reset to `None`.
                    task.take();
                }

                // Restore the back index to the original task.
                self.inner.back.store(b.wrapping_add(1), Ordering::Relaxed);
            } else {
                // Shrink the buffer if `len` is less than one fourth of the capacity.
                if buffer.cap > MIN_CAP && len < buffer.cap as isize / 4 {
                    unsafe {
                        self.resize(buffer.cap / 2);
                    }
                }
            }

            task.map(|t| unsafe { t.assume_init() })
        }
    }
}

impl<T> fmt::Debug for Worker<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad("Worker { .. }")
    }
}

/// A stealer handle of a worker queue.
///
/// Stealers can be shared among threads.
///
/// Task schedulers typically have a single worker queue per worker thread.
///
/// # Examples
///
/// ```
/// use crossbeam_deque::{Steal, Worker};
///
/// let w = Worker::new_lifo();
/// w.push(1);
/// w.push(2);
///
/// let s = w.stealer();
/// assert_eq!(s.steal(), Steal::Success(1));
/// assert_eq!(s.steal(), Steal::Success(2));
/// assert_eq!(s.steal(), Steal::Empty);
/// ```
pub struct Stealer<T> {
    /// A reference to the inner representation of the queue.
    inner: Arc<CachePadded<Inner<T>>>,
}

unsafe impl<T: Send> Send for Stealer<T> {}
unsafe impl<T: Send> Sync for Stealer<T> {}

impl<T> Stealer<T> {
    /// Returns `true` if the queue is empty.
    ///
    /// ```
    /// use crossbeam_deque::Worker;
    ///
    /// let w = Worker::new_lifo();
    /// let s = w.stealer();
    ///
    /// assert!(s.is_empty());
    /// w.push(1);
    /// assert!(!s.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        // BAZI TODO: the steal index has to be compared instead of front
       // semantics: are there stealable tasks?
        let f = self.inner.front.load(Ordering::Acquire);
        atomic::fence(Ordering::SeqCst);
        let b = self.inner.back.load(Ordering::Acquire);
        b.wrapping_sub(f) <= 0
    }

    /// Returns the number of tasks in the deque.
    ///
    /// ```
    /// use crossbeam_deque::Worker;
    ///
    /// let w = Worker::new_lifo();
    /// let s = w.stealer();
    ///
    /// assert_eq!(s.len(), 0);
    /// w.push(1);
    /// assert_eq!(s.len(), 1);
    /// w.push(2);
    /// assert_eq!(s.len(), 2);
    /// ```
    pub fn len(&self) -> usize {
        // BAZI TODO: the steal index has to be compared instead of front
        // semantics: nr of stealable tasks
        let f = self.inner.front.load(Ordering::Acquire);
        atomic::fence(Ordering::SeqCst);
        let b = self.inner.back.load(Ordering::Acquire);
        b.wrapping_sub(f).max(0) as usize
    }

    /// Steals a task from the queue.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbeam_deque::{Steal, Worker};
    ///
    /// let w = Worker::new_lifo();
    /// w.push(1);
    /// w.push(2);
    ///
    /// let s = w.stealer();
    /// assert_eq!(s.steal(), Steal::Success(1));
    /// assert_eq!(s.steal(), Steal::Success(2));
    /// ```
    pub fn steal(&self) -> Steal<T> {
        // BAZI TODO: load stealable index
        // Load the front index.
        let f = self.inner.front.load(Ordering::Acquire);

        // A SeqCst fence is needed here.
        //
        // If the current thread is already pinned (reentrantly), we must manually issue the
        // fence. Otherwise, the following pinning will issue the fence anyway, so we don't
        // have to.
        if epoch::is_pinned() {
            atomic::fence(Ordering::SeqCst);
        }

        let guard = &epoch::pin();

        // Load the back index.
        let b = self.inner.back.load(Ordering::Acquire);

        // BAZI: check semantics with wrapping
        // Is the queue empty?
        if b.wrapping_sub(f) <= 0 {
            return Steal::Empty;
        }

        // Load the buffer and read the task at the front.
        let buffer = self.inner.buffer.load(Ordering::Acquire, guard);
        let task = unsafe { buffer.deref().read(f) };

        // BAZI increment stealindex [f], 'front' should be unchanged
        // Try incrementing the front index to steal the task.
        // If the buffer has been swapped or the increment fails, we retry.
        if self.inner.buffer.load(Ordering::Acquire, guard) != buffer
            || self
                .inner
                .front
                .compare_exchange(f, f.wrapping_add(1), Ordering::SeqCst, Ordering::Relaxed)
                .is_err()
        {
            // We didn't steal this task, forget it.
            return Steal::Retry;
        }

        // Return the stolen task.
        Steal::Success(unsafe { task.assume_init() })
    }
}

impl<T> Clone for Stealer<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<T> fmt::Debug for Stealer<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad("Stealer { .. }")
    }
}

/// Possible outcomes of a steal operation.
///
/// # Examples
///
/// There are lots of ways to chain results of steal operations together:
///
/// ```
/// use crossbeam_deque::Steal::{self, Empty, Retry, Success};
///
/// let collect = |v: Vec<Steal<i32>>| v.into_iter().collect::<Steal<i32>>();
///
/// assert_eq!(collect(vec![Empty, Empty, Empty]), Empty);
/// assert_eq!(collect(vec![Empty, Retry, Empty]), Retry);
/// assert_eq!(collect(vec![Retry, Success(1), Empty]), Success(1));
///
/// assert_eq!(collect(vec![Empty, Empty]).or_else(|| Retry), Retry);
/// assert_eq!(collect(vec![Retry, Empty]).or_else(|| Success(1)), Success(1));
/// ```
#[must_use]
#[derive(PartialEq, Eq, Copy, Clone)]
pub enum Steal<T> {
    /// The queue was empty at the time of stealing.
    Empty,

    /// At least one task was successfully stolen.
    Success(T),

    /// The steal operation needs to be retried.
    Retry,
}

impl<T> Steal<T> {
    /// Returns `true` if the queue was empty at the time of stealing.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbeam_deque::Steal::{Empty, Retry, Success};
    ///
    /// assert!(!Success(7).is_empty());
    /// assert!(!Retry::<i32>.is_empty());
    ///
    /// assert!(Empty::<i32>.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        matches!(self, Self::Empty)
    }

    /// Returns `true` if at least one task was stolen.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbeam_deque::Steal::{Empty, Retry, Success};
    ///
    /// assert!(!Empty::<i32>.is_success());
    /// assert!(!Retry::<i32>.is_success());
    ///
    /// assert!(Success(7).is_success());
    /// ```
    pub fn is_success(&self) -> bool {
        matches!(self, Self::Success(_))
    }

    /// Returns `true` if the steal operation needs to be retried.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbeam_deque::Steal::{Empty, Retry, Success};
    ///
    /// assert!(!Empty::<i32>.is_retry());
    /// assert!(!Success(7).is_retry());
    ///
    /// assert!(Retry::<i32>.is_retry());
    /// ```
    pub fn is_retry(&self) -> bool {
        matches!(self, Self::Retry)
    }

    /// Returns the result of the operation, if successful.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbeam_deque::Steal::{Empty, Retry, Success};
    ///
    /// assert_eq!(Empty::<i32>.success(), None);
    /// assert_eq!(Retry::<i32>.success(), None);
    ///
    /// assert_eq!(Success(7).success(), Some(7));
    /// ```
    pub fn success(self) -> Option<T> {
        match self {
            Self::Success(res) => Some(res),
            _ => None,
        }
    }

    /// If no task was stolen, attempts another steal operation.
    ///
    /// Returns this steal result if it is `Success`. Otherwise, closure `f` is invoked and then:
    ///
    /// * If the second steal resulted in `Success`, it is returned.
    /// * If both steals were unsuccessful but any resulted in `Retry`, then `Retry` is returned.
    /// * If both resulted in `None`, then `None` is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbeam_deque::Steal::{Empty, Retry, Success};
    ///
    /// assert_eq!(Success(1).or_else(|| Success(2)), Success(1));
    /// assert_eq!(Retry.or_else(|| Success(2)), Success(2));
    ///
    /// assert_eq!(Retry.or_else(|| Empty), Retry::<i32>);
    /// assert_eq!(Empty.or_else(|| Retry), Retry::<i32>);
    ///
    /// assert_eq!(Empty.or_else(|| Empty), Empty::<i32>);
    /// ```
    pub fn or_else<F>(self, f: F) -> Self
    where
        F: FnOnce() -> Self,
    {
        match self {
            Self::Empty => f(),
            Self::Success(_) => self,
            Self::Retry => {
                if let Self::Success(res) = f() {
                    Self::Success(res)
                } else {
                    Self::Retry
                }
            }
        }
    }
}

impl<T> fmt::Debug for Steal<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => f.pad("Empty"),
            Self::Success(_) => f.pad("Success(..)"),
            Self::Retry => f.pad("Retry"),
        }
    }
}
