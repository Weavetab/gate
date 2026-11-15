use std::collections::{HashMap, VecDeque};

pub(crate) struct RingCache<V> {
    cap: usize,
    order: VecDeque<u64>,
    map: HashMap<u64, V>,
}

impl<V: Clone> RingCache<V> {
    pub fn new(cap: usize) -> Self {
        Self {
            cap: cap.max(1),
            order: VecDeque::new(),
            map: HashMap::new(),
        }
    }

    pub fn get(&self, key: u64) -> Option<V> {
        self.map.get(&key).cloned()
    }

    pub fn put(&mut self, key: u64, val: V) {
        if self.map.contains_key(&key) {
            self.map.insert(key, val);
            return;
        }
        if self.order.len() >= self.cap {
            if let Some(old) = self.order.pop_front() {
                self.map.remove(&old);
            }
        }
        self.order.push_back(key);
        self.map.insert(key, val);
    }
}

// fnv-1a 64-bit so cache keys need zero extra hashing crates
pub(crate) fn hash_key(tag: u8, payload: &[u8], choices: &[&str]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    let mix = |h: &mut u64, bytes: &[u8]| {
        for &b in bytes {
            *h ^= b as u64;
            *h = h.wrapping_mul(0x100000001b3);
        }
        *h ^= 0xff;
        *h = h.wrapping_mul(0x100000001b3);
    };
    mix(&mut h, &[tag]);
    mix(&mut h, payload);
    for c in choices {
        mix(&mut h, c.as_bytes());
    }
    h
}
