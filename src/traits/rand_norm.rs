use std::num::Wrapping;

pub trait RandNorm {
    type SeedType;
    fn rand_next(seed: &mut Self::SeedType) -> Self;
}

impl RandNorm for f32 {
    type SeedType = u64;

    fn rand_next(seed: &mut u64) -> Self {
        *seed = (Wrapping(2074984187 as u64) * Wrapping(*seed) + Wrapping(2881137594 as u64)).0;
        (*seed >> 32) as f32 / (u32::MAX as f32)
    }
}

impl RandNorm for f64 {
    type SeedType = u64;

    fn rand_next(seed: &mut u64) -> Self {
        u64::rand_next(seed) as f64  / (u64::MAX as f64)
    }
}

impl RandNorm for i32 {
    type SeedType = u64;

    fn rand_next(seed: &mut u64) -> Self {
        *seed = (Wrapping(2074984187 as u64) * Wrapping(*seed) + Wrapping(2881137594 as u64)).0;
        (*seed >> 32) as i32
    }
}

impl RandNorm for u32 {
    type SeedType = u64;

    fn rand_next(seed: &mut u64) -> Self {
        *seed = (Wrapping(2074984187 as u64) * Wrapping(*seed) + Wrapping(2881137594 as u64)).0;
        (*seed >> 32) as u32
    }
}

impl RandNorm for u64 {
    type SeedType = u64;

    fn rand_next(seed: &mut u64) -> Self {
        let r0 = u32::rand_next(seed);
        let r1 = u32::rand_next(seed);
        ((r0 as u64) << 32) | (r1 as u64)
    }
}

impl RandNorm for i64 {
    type SeedType = u64;

    fn rand_next(seed: &mut u64) -> Self {
        let r0 = u32::rand_next(seed);
        let r1 = u32::rand_next(seed);
        ((r0 as i64) << 32) | (r1 as i64)
    }
}