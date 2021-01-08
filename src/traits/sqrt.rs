pub trait Sqrt {
    type Output;
    fn sqrt(&self) -> Self::Output;
}

impl Sqrt for f32 {
    type Output = Self;
    fn sqrt(&self) -> Self::Output {
        f32::sqrt(*self)
    }
}

impl Sqrt for f64 {
    type Output = Self;
    fn sqrt(&self) -> Self::Output {
        f64::sqrt(*self)
    }
}