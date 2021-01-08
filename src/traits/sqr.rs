pub trait Sqr {
    fn sqr(&self) -> Self;
}

impl Sqr for f32 {
    fn sqr(&self) -> Self {
        self * self
    }
}

impl Sqr for f64 {
    fn sqr(&self) -> Self {
        self * self
    }
}

impl Sqr for i8 {
    fn sqr(&self) -> Self {
        self * self
    }
}

impl Sqr for i16 {
    fn sqr(&self) -> Self {
        self * self
    }
}

impl Sqr for i32 {
    fn sqr(&self) -> Self {
        self * self
    }
}

impl Sqr for i64 {
    fn sqr(&self) -> Self {
        self * self
    }
}

impl Sqr for u8 {
    fn sqr(&self) -> Self {
        self * self
    }
}

impl Sqr for u16 {
    fn sqr(&self) -> Self {
        self * self
    }
}

impl Sqr for u32 {
    fn sqr(&self) -> Self {
        self * self
    }
}

impl Sqr for u64 {
    fn sqr(&self) -> Self {
        self * self
    }
}