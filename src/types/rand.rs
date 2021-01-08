use crate::RandNorm;

pub struct Rand<T: RandNorm> {
    seed: T::SeedType,
}

impl<T: RandNorm> Rand<T> {
    pub fn new(seed: T::SeedType) -> Self {
        Self {
            seed,
        }
    }

    pub fn next(&mut self) -> T {
        T::rand_next(&mut self.seed)
    }
}