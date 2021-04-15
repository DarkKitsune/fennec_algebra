use std::num::Wrapping;

pub trait Seed {
    fn seed(&self) -> u64;
}

impl Seed for i8 {
    fn seed(&self) -> u64 {
        *self as u64
    }
}

impl Seed for i16 {
    fn seed(&self) -> u64 {
        *self as u64
    }
}

impl Seed for i32 {
    fn seed(&self) -> u64 {
        *self as u64
    }
}

impl Seed for i64 {
    fn seed(&self) -> u64 {
        *self as u64
    }
}

impl Seed for isize {
    fn seed(&self) -> u64 {
        *self as u64
    }
}

impl Seed for u8 {
    fn seed(&self) -> u64 {
        *self as u64
    }
}

impl Seed for u16 {
    fn seed(&self) -> u64 {
        *self as u64
    }
}

impl Seed for u32 {
    fn seed(&self) -> u64 {
        *self as u64
    }
}

impl Seed for u64 {
    fn seed(&self) -> u64 {
        *self
    }
}

impl Seed for usize {
    fn seed(&self) -> u64 {
        *self as u64
    }
}

impl Seed for f32 {
    fn seed(&self) -> u64 {
        unsafe { *(self as *const f32 as *const u32) as u64 }
    }
}

impl Seed for f64 {
    fn seed(&self) -> u64 {
        unsafe { *(self as *const f64 as *const u64) }
    }
}

impl<T: Seed, const SIZE: usize> Seed for [T; SIZE] {
    fn seed(&self) -> u64 {
        let mut seed0 = 9437275731793346971;
        let mut seed1 = 12507178061862636611;
        for e in self.iter() {
            seed0 ^= (Wrapping(e.seed()) * Wrapping(12507178061862636611)).0;
            seed1 ^= (Wrapping(e.seed()) * Wrapping(9437275731793346971)).0;
        }
        seed0 << 32 | ((seed1 << 32) >> 32)
    }
}

impl<T: Seed> Seed for Vec<T> {
    fn seed(&self) -> u64 {
        let mut seed0 = 9437275731793346971;
        let mut seed1 = 12507178061862636611;
        for e in self.iter() {
            seed0 ^= (Wrapping(e.seed()) * Wrapping(12507178061862636611)).0;
            seed1 ^= (Wrapping(e.seed()) * Wrapping(9437275731793346971)).0;
        }
        seed0 << 32 | ((seed1 << 32) >> 32)
    }
}
