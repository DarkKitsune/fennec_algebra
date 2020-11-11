use std::iter::Sum;
use std::ops::Mul;

pub trait Dot<T>: Iterator<Item = (T, T)> + Sized
where
    T: Mul,
    <T as Mul>::Output: Sum,
{
    fn dot(self) -> <T as Mul>::Output {
        self.into_iter().map(|(a, b)| a * b).sum()
    }
}

impl<T, I> Dot<I> for T
where
    T: Iterator<Item = (I, I)>,
    I: Mul,
    <I as Mul>::Output: Sum,
{
}
