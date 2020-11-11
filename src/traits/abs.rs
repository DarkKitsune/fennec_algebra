pub trait Abs {
    type Output;
    fn abs(&self) -> Self::Output;
}

impl Abs for f32 {
    type Output = Self;
    fn abs(&self) -> Self::Output {
        f32::abs(*self)
    }
}

impl Abs for f64 {
    type Output = Self;
    fn abs(&self) -> Self::Output {
        f64::abs(*self)
    }
}
