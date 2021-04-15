use crate::*;

#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct Quaternion {
    components: Vector<f32, 4>,
}

impl Quaternion {
    pub const IDENTITY: Self = Self::new(0.0, 0.0, 0.0, 1.0);

    pub const fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self {
            components: vector!(x, y, z, w),
        }
    }

    pub fn from_vec(xyz: Vector<f32, 3>, w: f32) -> Self {
        Self {
            components: vector!(xyz[0], xyz[1], xyz[2], w),
        }
    }

    pub fn from_vec4(xyzw: Vector<f32, 4>) -> Self {
        Self { components: xyzw }
    }

    pub fn from_axis_angle(axis: Vector<f32, 3>, radians: f32) -> Result<Self, &'static str> {
        let normal = axis
            .normalized()
            .map_err(|_| "Axis has a length of 0 or close to 0")?;
        let half_angle = radians * 0.5;
        Ok(
            Quaternion::from_vec(normal * half_angle.sin(), half_angle.cos())
                .normalized()
                .unwrap(),
        )
    }

    pub fn x(&self) -> f32 {
        *self.components.x()
    }

    pub fn y(&self) -> f32 {
        *self.components.y()
    }

    pub fn z(&self) -> f32 {
        *self.components.z()
    }

    pub fn w(&self) -> f32 {
        *self.components.w()
    }

    pub fn xyz(&self) -> Vector<f32, 3> {
        vector!(self.components[0], self.components[1], self.components[2])
    }

    pub fn xyzw(&self) -> Vector<f32, 4> {
        self.components
    }

    pub fn set_x(&mut self, val: f32) {
        self.components[0] = val;
    }

    pub fn set_y(&mut self, val: f32) {
        self.components[1] = val;
    }

    pub fn set_z(&mut self, val: f32) {
        self.components[2] = val;
    }

    pub fn set_w(&mut self, val: f32) {
        self.components[3] = val;
    }

    pub fn set_xyz(&mut self, val: Vector<f32, 3>) {
        self.components[0] = val[0];
        self.components[1] = val[1];
        self.components[2] = val[2];
    }

    pub fn length(&self) -> f32 {
        self.length_squared().sqrt()
    }

    pub fn length_squared(&self) -> f32 {
        self.w() * self.w() + self.xyz().length2().unwrap()
    }

    pub fn normalized(&self) -> Result<Quaternion, QuaternionError> {
        let len_sq = self.length_squared();
        if len_sq < 0.00001 {
            Err(QuaternionError::ZeroLength)
        } else {
            let len = len_sq.sqrt();
            Ok(Quaternion::from_vec(self.xyz() / len, self.w() / len))
        }
    }

    pub fn invert(&mut self) -> Quaternion {
        let len_sq = self.length_squared();
        if len_sq.abs() > 0.00001 {
            let i = 1.0 / len_sq;
            Quaternion::from_vec(self.xyz() * -i, self.w() * i)
        } else {
            *self
        }
    }

    pub fn axis_angle(&self) -> (Vector<f32, 3>, f32) {
        let norm = if self.w() > 1.0 {
            self.normalized().unwrap()
        } else {
            *self
        };
        let w = 2.0 * norm.w().acos();
        let den = (1.0 - norm.w() * norm.w()).sqrt();
        if den > 0.00001 {
            (norm.xyz() / den, w)
        } else {
            (vector!(1.0, 0.0, 0.0), 0.0)
        }
    }

    pub fn conjugate(&self) -> Self {
        Quaternion::from_vec(vector!(0.0, 0.0, 0.0) - self.xyz(), self.w())
    }
}

impl std::ops::Add<Self> for Quaternion {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self::from_vec4(self.xyzw() + other.xyzw())
    }
}

impl std::ops::Add<f32> for Quaternion {
    type Output = Self;

    fn add(self, other: f32) -> Self {
        Self::from_vec4(self.xyzw() + other)
    }
}

impl std::ops::Sub<Self> for Quaternion {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self::from_vec4(self.xyzw() - other.xyzw())
    }
}

impl std::ops::Sub<f32> for Quaternion {
    type Output = Self;

    fn sub(self, other: f32) -> Self {
        Self::from_vec4(self.xyzw() - other)
    }
}

impl std::ops::Mul<Self> for Quaternion {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        let cross = self.xyz().cross(&other.xyz());
        let dot = self.xyz().dot(&other.xyz());
        Self::new(
            self.x() * other.w() + other.x() * self.w() + cross.x(),
            self.y() * other.w() + other.y() * self.w() + cross.y(),
            self.z() * other.w() + other.z() * self.w() + cross.z(),
            self.w() * other.w() - dot,
        )
    }
}

impl std::ops::Mul<f32> for Quaternion {
    type Output = Self;

    fn mul(self, other: f32) -> Self {
        Self::from_vec4(self.xyzw() * other)
    }
}

impl std::ops::Div<Self> for Quaternion {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        Self::from_vec4(self.xyzw() / other.xyzw())
    }
}

impl std::ops::Div<f32> for Quaternion {
    type Output = Self;

    fn div(self, other: f32) -> Self {
        Self::from_vec4(self.xyzw() / other)
    }
}

impl std::ops::Rem<Self> for Quaternion {
    type Output = Self;

    fn rem(self, other: Self) -> Self {
        Self::from_vec4(self.xyzw() % other.xyzw())
    }
}

impl std::ops::Rem<f32> for Quaternion {
    type Output = Self;

    fn rem(self, other: f32) -> Self {
        Self::from_vec4(self.xyzw() % other)
    }
}

impl std::ops::Neg for Quaternion {
    type Output = Self;

    fn neg(self) -> Self {
        Self::from_vec4(vector!(0.0, 0.0, 0.0, 0.0) - self.xyzw())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Hash)]
pub enum QuaternionError {
    ZeroLength,
    VectorError(VectorError),
}
