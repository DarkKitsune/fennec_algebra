use crate::*;
use std::num::Wrapping;

pub struct ValRand {
    base_seed: u64,
}

impl ValRand {
    pub fn new<S: Seed>(base_seed: S) -> Self {
        Self {
            base_seed: base_seed.seed(),
        }
    }

    pub fn next<T: RandNorm<SeedType = u64>, S: Seed>(&self, seed: S) -> T {
        let seed =
            Wrapping(seed.seed()) * Wrapping(16339635446596153441) + Wrapping(self.base_seed);
        let seed0 = seed.0 << 32;
        let seed = seed * Wrapping(16339635446596153441) + Wrapping(10217985274814629069);
        let seed1 = (seed.0 << 32) >> 32;
        RandNorm::rand_next(&mut (seed0 | seed1))
    }
}

pub trait ToRand: Seed {
    fn to_rand<T: RandNorm<SeedType = u64>>(&self) -> T {
        let seed =
            Wrapping(self.seed()) * Wrapping(16339635446596153441) + Wrapping(10217985274814629069);
        let seed0 = seed.0 << 32;
        let seed = seed * Wrapping(16339635446596153441) + Wrapping(10217985274814629069);
        let seed1 = (seed.0 << 32) >> 32;
        RandNorm::rand_next(&mut (seed0 | seed1))
    }
}

impl<T: Seed> ToRand for T {}
