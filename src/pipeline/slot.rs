use std::sync::Mutex;
use tokio::sync::Notify;

pub struct LastSlot<T> {
    inner:  Mutex<Option<T>>,
    notify: Notify,
}

impl<T: Send> LastSlot<T> {
    pub fn new() -> Self {
        Self { inner: Mutex::new(None), notify: Notify::new() }
    }

    /// Swap in `val`; return any displaced predecessor.
    pub fn put(&self, val: T) -> Option<T> {
        let old = self.inner.lock().unwrap().replace(val);
        self.notify.notify_one();
        old
    }

    /// Suspend until a value is available, then take it.
    pub async fn take(&self) -> T {
        loop {
            {
                let mut guard = self.inner.lock().unwrap();
                if let Some(val) = guard.take() {
                    return val;
                }
            }
            self.notify.notified().await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_take_after_put() {
        let slot = LastSlot::new();
        slot.put(42u32);
        assert_eq!(slot.take().await, 42);
    }

    #[tokio::test]
    async fn test_put_displaces_previous() {
        let slot = LastSlot::new();
        assert!(slot.put(1u32).is_none());   // slot was empty
        assert_eq!(slot.put(2u32), Some(1)); // previous value returned
        assert_eq!(slot.take().await, 2);    // latest value wins
    }

    #[tokio::test]
    async fn test_take_blocks_until_put() {
        use std::sync::Arc;
        use tokio::time::{sleep, Duration};

        let slot = Arc::new(LastSlot::new());
        let slot2 = Arc::clone(&slot);
        let handle = tokio::spawn(async move { slot2.take().await });
        sleep(Duration::from_millis(10)).await;
        slot.put(99u32);
        assert_eq!(handle.await.unwrap(), 99);
    }
}
