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
        *seed = (Wrapping(2074984187 as u64) * Wrapping(*seed) + Wrapping(2881137594 as u64)).0;
        (*seed >> 32) as f64 / (u32::MAX as f64)
    }
}
