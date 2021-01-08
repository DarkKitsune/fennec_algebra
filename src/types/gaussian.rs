use crate::{RandNorm, One};
use std::ops::{Add, Mul};

pub struct Gaussian2<T: RandNorm + One + Clone + Mul<T, Output = T> + Add<T, Output = T> + PartialOrd + Sized> {
    seed: T::SeedType,
    next: Option<T>,
}

impl<T: RandNorm + One + Clone + Mul<T, Output = T> + Add<T, Output = T> + PartialOrd + Sized> Gaussian2<T> {
    pub fn new(seed: T::SeedType) -> Self {
        Self {
            seed,
            next: None,
        }
    }

    pub fn next(&mut self) -> T {
        if self.next.is_some() {
            let mut next = None;
            std::mem::swap(&mut next, &mut self.next);
            next.unwrap()
        }
        else {
            loop {
                let a = T::rand_next(&mut self.seed);
                let b = T::rand_next(&mut self.seed);
                if a.clone() * a.clone() + b.clone() * b.clone() < T::one() {
                    let mut next = Some(b);
                    std::mem::swap(&mut next, &mut self.next);
                    return a;
                }
            }
        }
    }
}

pub struct Gaussian3<T: RandNorm + One + Clone + Mul<T, Output = T> + Add<T, Output = T> + PartialOrd + Sized> {
    seed: T::SeedType,
    next0: Option<T>,
    next1: Option<T>,
}

impl<T: RandNorm + One + Clone + Mul<T, Output = T> + Add<T, Output = T> + PartialOrd + Sized> Gaussian3<T> {
    pub fn new(seed: T::SeedType) -> Self {
        Self {
            seed,
            next0: None,
            next1: None,
        }
    }

    pub fn next(&mut self) -> T {
        if self.next0.is_some() {
            let mut next = None;
            std::mem::swap(&mut next, &mut self.next0);
            next.unwrap()
        }
        else if self.next1.is_some() {
            let mut next = None;
            std::mem::swap(&mut next, &mut self.next1);
            next.unwrap()
        }
        else {
            loop {
                let a = T::rand_next(&mut self.seed);
                let b = T::rand_next(&mut self.seed);
                let c = T::rand_next(&mut self.seed);
                if a.clone() * a.clone() + b.clone() * b.clone() + c.clone() * c.clone() < T::one() {
                    let mut next0 = Some(b);
                    let mut next1 = Some(c);
                    std::mem::swap(&mut next0, &mut self.next0);
                    std::mem::swap(&mut next1, &mut self.next1);
                    return a;
                }
            }
        }
    }

    pub fn seed(&self) -> &T::SeedType {
        &self.seed
    }
}