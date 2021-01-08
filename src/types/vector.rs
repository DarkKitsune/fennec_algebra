use std::cmp::{Eq, PartialEq};
use std::fmt::{Debug, Formatter};
use std::hash::{Hash, Hasher};
use std::iter::Sum;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Rem, RemAssign, Sub, SubAssign, Neg,
};

use crate::{init_array, Sqrt, Zero, One, Two};

#[repr(C)]
pub struct Vector<T: Sized, const N: usize> {
    pub components: [T; N],
}

//type TruncatedVector<T, const N: usize> = Vector<T, {N - 1}>;

impl<T: Sized, const N: usize> Vector<T, N> {
    pub const fn new(components: [T; N]) -> Self
    where
        T: Sized,
    {
        Self { components }
    }

    pub fn length2(&self) -> Result<T, VectorError>
    where
        T: Add<T, Output = T> + Mul<T, Output = T> + Clone,
    {
        let mut components_iter = self.components.iter();
        let first = components_iter.next().ok_or(VectorError::ZeroComponents)?;
        let mut sum = first.clone() * first.clone();
        for x in components_iter {
            sum = sum + x.clone() * x.clone();
        }
        Ok(sum)
    }

    pub fn length<TSqrt>(&self) -> Result<TSqrt, VectorError>
    where
        T: Add<T, Output = T> + Mul<T, Output = T> + Sqrt<Output = TSqrt> + Clone,
    {
        Ok(self.length2()?.sqrt())
    }

    pub fn normalized(&self) -> Result<Self, VectorError>
    where
        T: Add<T, Output = T> + Sqrt<Output = T> + Clone,
        T: Mul<T, Output = T>,
        T: Div<T, Output = T>,
    {
        let length = self.length()?;
        Ok(Self::new(init_array!([T; N], |idx| {
            let component: &T = &self.components[idx];
            component.clone() / length.clone()
        })))
    }

    pub const fn component(&self, idx: usize) -> Result<&T, VectorError> {
        if idx < N {
            Ok(&self.components[idx])
        } else {
            Err(VectorError::NoComponentWithGivenIndex)
        }
    }

    pub fn component_mut(&mut self, idx: usize) -> Result<&mut T, VectorError> {
        if idx < N {
            Ok(&mut self.components[idx])
        } else {
            Err(VectorError::NoComponentWithGivenIndex)
        }
    }

    pub fn add_vector(&mut self, other: &Self)
    where
        T: Add<T, Output = T> + Clone,
    {
        for idx in 0..N {
            let a = self.components[idx].clone();
            let b = other.components[idx].clone();
            self.components[idx] = a + b;
        }
    }

    pub fn sub_vector(&mut self, other: &Self)
    where
        T: Sub<T, Output = T> + Clone,
    {
        for idx in 0..N {
            let a = self.components[idx].clone();
            let b = other.components[idx].clone();
            self.components[idx] = a - b;
        }
    }

    pub fn mul_vector(&mut self, other: &Self)
    where
        T: Mul<T, Output = T> + Clone,
    {
        for idx in 0..N {
            let a = self.components[idx].clone();
            let b = other.components[idx].clone();
            self.components[idx] = a * b;
        }
    }

    pub fn div_vector(&mut self, other: &Self)
    where
        T: Div<T, Output = T> + Clone,
    {
        for idx in 0..N {
            let a = self.components[idx].clone();
            let b = other.components[idx].clone();
            self.components[idx] = a / b;
        }
    }

    pub fn rem_vector(&mut self, other: &Self)
    where
        T: Rem<T, Output = T> + Clone,
    {
        for idx in 0..N {
            let a = self.components[idx].clone();
            let b = other.components[idx].clone();
            self.components[idx] = a % b;
        }
    }

    pub fn add_component(&mut self, other: &T)
    where
        T: Add<T, Output = T> + Clone,
    {
        for idx in 0..N {
            let a = self.components[idx].clone();
            let b = other.clone();
            self.components[idx] = a + b;
        }
    }

    pub fn sub_component(&mut self, other: &T)
    where
        T: Sub<T, Output = T> + Clone,
    {
        for idx in 0..N {
            let a = self.components[idx].clone();
            let b = other.clone();
            self.components[idx] = a - b;
        }
    }

    pub fn mul_component(&mut self, other: &T)
    where
        T: Mul<T, Output = T> + Clone,
    {
        for idx in 0..N {
            let a = self.components[idx].clone();
            let b = other.clone();
            self.components[idx] = a * b;
        }
    }

    pub fn div_component(&mut self, other: &T)
    where
        T: Div<T, Output = T> + Clone,
    {
        for idx in 0..N {
            let a = self.components[idx].clone();
            let b = other.clone();
            self.components[idx] = a / b;
        }
    }

    pub fn rem_component(&mut self, other: &T)
    where
        T: Rem<T, Output = T> + Clone,
    {
        for idx in 0..N {
            let a = self.components[idx].clone();
            let b = other.clone();
            self.components[idx] = a % b;
        }
    }
    /*
    pub fn truncate(&self) -> TruncatedVector<T, N>
        where TruncatedVector<T, N>: Clone,
        T: Clone,
    {
        Vector {
            components: init_array!([T; N - 1], |idx| {
                let component: &T = &self.components[idx];
                component.clone()
            }),
        }
    }*/

    pub fn dot(&self, other: &Self) -> T
    where
        T: Mul<T, Output = T> + Clone + Sum,
    {
        self.components
            .iter()
            .enumerate()
            .map(|(idx, component)| component.clone() * other.components[idx].clone())
            .sum()
    }

    pub fn convert<T2: Sized>(&self) -> Vector<T2, N>
    where
        T: Into<T2> + Clone,
    {
        Vector::new(init_array!([T2; N], |idx| {
            let component: &T = &self.components[idx];
            component.clone().into()
        }))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Hash)]
pub enum VectorError {
    ZeroComponents,
    NoComponentWithGivenIndex,
}

#[macro_export]
macro_rules! vector {
    ($($component:expr),+$(,)?) => {
        $crate::Vector::new([$($component),+])
    };
}

impl<T: Sized, const N: usize> Clone for Vector<T, N>
where
    T: Clone,
{
    fn clone(&self) -> Self {
        Self::new(init_array!([T; N], |idx| (&self.components[idx] as &T).clone()))
    }
}

impl<T: Sized, const N: usize> Copy for Vector<T, N> where T: Copy {}

impl<T: Sized, const N: usize> Debug for Vector<T, N>
where
    T: Debug,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "(")?;
        for idx in 0..N {
            if idx > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{:?}", &self.components[idx])?;
        }
        write!(f, ")")?;
        Ok(())
    }
}

impl<T: Sized, const N: usize> PartialEq<Vector<T, N>> for Vector<T, N>
where
    T: PartialEq<T>,
{
    fn eq(&self, rhs: &Vector<T, N>) -> bool {
        for idx in 0..N {
            if self.components[idx] != rhs.components[idx] {
                return false;
            }
        }
        true
    }
}

impl<T: Sized, const N: usize> Eq for Vector<T, N> where T: Eq {}

impl<T: Sized, const N: usize> Hash for Vector<T, N>
where
    T: Hash,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        for idx in 0..N {
            self.components[idx].hash(state);
        }
    }
}

impl<T: Sized, const N: usize> Default for Vector<T, N>
where
    T: Default,
{
    fn default() -> Self {
        Vector::new(init_array!([T; N], |_| T::default()))
    }
}

impl<T: Sized, const N: usize> Zero for Vector<T, N>
where
    T: Zero,
{
    fn zero() -> Self {
        Vector::new(init_array!([T; N], |_| T::zero()))
    }
}

impl<T: Sized, const N: usize> One for Vector<T, N>
where
    T: One,
{
    fn one() -> Self {
        Vector::new(init_array!([T; N], |_| T::one()))
    }
}

impl<T: Sized, const N: usize> Two for Vector<T, N>
where
    T: Two,
{
    fn two() -> Self {
        Vector::new(init_array!([T; N], |_| T::two()))
    }
}

impl<T: Sized, const N: usize> Neg for Vector<T, N>
where
    T: Neg + Clone,
{
    type Output = Vector<<T as Neg>::Output, N>;

    fn neg(self) -> Vector<<T as Neg>::Output, N> {
        Vector::new(init_array!([<T as Neg>::Output; N], |idx| -self[idx].clone()))
    }
}

impl<T: Sized, const N: usize> Neg for &Vector<T, N>
where
    T: Neg + Clone,
{
    type Output = Vector<<T as Neg>::Output, N>;

    fn neg(self) -> Vector<<T as Neg>::Output, N> {
        Vector::new(init_array!([<T as Neg>::Output; N], |idx| -self[idx].clone()))
    }
}

impl<T: Sized, const N: usize> Neg for &mut Vector<T, N>
where
    T: Neg + Clone,
{
    type Output = Vector<<T as Neg>::Output, N>;

    fn neg(self) -> Vector<<T as Neg>::Output, N> {
        Vector::new(init_array!([<T as Neg>::Output; N], |idx| -self[idx].clone()))
    }
}

impl<T: Sized, const N: usize> Index<usize> for Vector<T, N> {
    type Output = T;

    fn index(&self, idx: usize) -> &T {
        &self.components[idx]
    }
}

impl<T: Sized, const N: usize> IndexMut<usize> for Vector<T, N> {
    fn index_mut(&mut self, idx: usize) -> &mut T {
        &mut self.components[idx]
    }
}

macro_rules! vector_binary_op {
    ($op:ident, $fn_name:ident, $type_method_component:ident, $type_method:ident) => {
        impl<'a, 'b, T: Sized, const N: usize> $op<&'b Vector<T, N>> for &'a Vector<T, N>
        where
            T: $op<T, Output = T> + Clone,
            Vector<T, N>: Clone,
        {
            type Output = Vector<T, N>;

            fn $fn_name(self, rhs: &'b Vector<T, N>) -> Vector<T, N> {
                let mut new = self.clone();
                new.$type_method(rhs);
                new
            }
        }
        impl<'a, 'b, T: Sized, const N: usize> $op<&'b Vector<T, N>> for &'a mut Vector<T, N>
        where
            T: $op<T, Output = T> + Clone,
            Vector<T, N>: Clone,
        {
            type Output = Vector<T, N>;

            fn $fn_name(self, rhs: &'b Vector<T, N>) -> Vector<T, N> {
                let mut new = self.clone();
                new.$type_method(rhs);
                new
            }
        }

        impl<'a, T: Sized, const N: usize> $op<&'a Vector<T, N>> for Vector<T, N>
        where
            T: $op<T, Output = T> + Clone,
            Vector<T, N>: Clone,
        {
            type Output = Vector<T, N>;

            fn $fn_name(self, rhs: &'a Vector<T, N>) -> Vector<T, N> {
                let mut new = self.clone();
                new.$type_method(rhs);
                new
            }
        }

        impl<'a, T: Sized, const N: usize> $op<Vector<T, N>> for &'a Vector<T, N>
        where
            T: $op<T, Output = T> + Clone,
            Vector<T, N>: Clone,
        {
            type Output = Vector<T, N>;

            fn $fn_name(self, rhs: Vector<T, N>) -> Vector<T, N> {
                let mut new = self.clone();
                new.$type_method(&rhs);
                new
            }
        }

        impl<'a, T: Sized, const N: usize> $op<Vector<T, N>> for &'a mut Vector<T, N>
        where
            T: $op<T, Output = T> + Clone,
            Vector<T, N>: Clone,
        {
            type Output = Vector<T, N>;

            fn $fn_name(self, rhs: Vector<T, N>) -> Vector<T, N> {
                let mut new = self.clone();
                new.$type_method(&rhs);
                new
            }
        }

        impl<T: Sized, const N: usize> $op<Vector<T, N>> for Vector<T, N>
        where
            T: $op<T, Output = T> + Clone,
            Vector<T, N>: Clone,
        {
            type Output = Vector<T, N>;

            fn $fn_name(self, rhs: Vector<T, N>) -> Vector<T, N> {
                let mut new = self.clone();
                new.$type_method(&rhs);
                new
            }
        }

        impl<'a, 'b, T: Sized, const N: usize> $op<&'b T> for &'a Vector<T, N>
        where
            T: $op<T, Output = T> + Clone,
            Vector<T, N>: Clone,
        {
            type Output = Vector<T, N>;

            fn $fn_name(self, rhs: &'b T) -> Vector<T, N> {
                let mut new = self.clone();
                new.$type_method_component(rhs);
                new
            }
        }

        impl<'a, 'b, T: Sized, const N: usize> $op<&'b T> for &'a mut Vector<T, N>
        where
            T: $op<T, Output = T> + Clone,
            Vector<T, N>: Clone,
        {
            type Output = Vector<T, N>;

            fn $fn_name(self, rhs: &'b T) -> Vector<T, N> {
                let mut new = self.clone();
                new.$type_method_component(rhs);
                new
            }
        }

        impl<'a, T: Sized, const N: usize> $op<&'a T> for Vector<T, N>
        where
            T: $op<T, Output = T> + Clone,
            Vector<T, N>: Clone,
        {
            type Output = Vector<T, N>;

            fn $fn_name(self, rhs: &'a T) -> Vector<T, N> {
                let mut new = self.clone();
                new.$type_method_component(rhs);
                new
            }
        }

        impl<'a, T: Sized, const N: usize> $op<T> for &'a Vector<T, N>
        where
            T: $op<T, Output = T> + Clone,
            Vector<T, N>: Clone,
        {
            type Output = Vector<T, N>;

            fn $fn_name(self, rhs: T) -> Vector<T, N> {
                let mut new = self.clone();
                new.$type_method_component(&rhs);
                new
            }
        }

        impl<'a, T: Sized, const N: usize> $op<T> for &'a mut Vector<T, N>
        where
            T: $op<T, Output = T> + Clone,
            Vector<T, N>: Clone,
        {
            type Output = Vector<T, N>;

            fn $fn_name(self, rhs: T) -> Vector<T, N> {
                let mut new = self.clone();
                new.$type_method_component(&rhs);
                new
            }
        }

        impl<T: Sized, const N: usize> $op<T> for Vector<T, N>
        where
            T: $op<T, Output = T> + Clone,
            Vector<T, N>: Clone,
        {
            type Output = Vector<T, N>;

            fn $fn_name(self, rhs: T) -> Vector<T, N> {
                let mut new = self.clone();
                new.$type_method_component(&rhs);
                new
            }
        }
    };
}

macro_rules! vector_assign_op {
    ($op:ident, $op_assign:ident, $fn_name:ident, $type_method_component:ident, $type_method:ident) => {
        impl<T: Sized, const N: usize> $op_assign<Vector<T, N>> for Vector<T, N>
        where
            T: $op<T, Output = T> + Clone,
        {
            fn $fn_name(&mut self, other: Self) {
                self.$type_method(&other);
            }
        }

        impl<T: Sized, const N: usize> $op_assign<&Vector<T, N>> for Vector<T, N>
        where
            T: $op<T, Output = T> + Clone,
        {
            fn $fn_name(&mut self, other: &Self) {
                self.$type_method(other);
            }
        }

        impl<T: Sized, const N: usize> $op_assign<&T> for Vector<T, N>
        where
            T: $op<T, Output = T> + Clone,
        {
            fn $fn_name(&mut self, other: &T) {
                self.$type_method_component(other);
            }
        }

        impl<T: Sized, const N: usize> $op_assign<T> for Vector<T, N>
        where
            T: $op<T, Output = T> + Clone,
        {
            fn $fn_name(&mut self, other: T) {
                self.$type_method_component(&other);
            }
        }
    };
}

vector_binary_op!(Add, add, add_component, add_vector);
vector_binary_op!(Sub, sub, sub_component, sub_vector);
vector_binary_op!(Mul, mul, mul_component, mul_vector);
vector_binary_op!(Div, div, div_component, div_vector);
vector_binary_op!(Rem, rem, rem_component, rem_vector);
vector_assign_op!(Add, AddAssign, add_assign, add_component, add_vector);
vector_assign_op!(Sub, SubAssign, sub_assign, sub_component, sub_vector);
vector_assign_op!(Mul, MulAssign, mul_assign, mul_component, mul_vector);
vector_assign_op!(Div, DivAssign, div_assign, div_component, div_vector);
vector_assign_op!(Rem, RemAssign, rem_assign, rem_component, rem_vector);

pub trait VectorX<T> {
    fn x(&self) -> &T;
    fn x_mut(&mut self) -> &mut T;
}

pub trait VectorXY<T> {
    fn x(&self) -> &T;
    fn x_mut(&mut self) -> &mut T;
    fn y(&self) -> &T;
    fn y_mut(&mut self) -> &mut T;
}

pub trait VectorXYZ<T> {
    fn x(&self) -> &T;
    fn x_mut(&mut self) -> &mut T;
    fn y(&self) -> &T;
    fn y_mut(&mut self) -> &mut T;
    fn z(&self) -> &T;
    fn z_mut(&mut self) -> &mut T;
    fn xy(&self) -> Vector<T, 2>
        where T: Clone;
    fn yz(&self) -> Vector<T, 2>
        where T: Clone;

    fn cross(&self, other: &Self) -> Vector<T, 3>
        where T: Mul<T, Output = T> + Sub<T, Output = T> + Clone
    {
        vector!(
            self.y().clone() * other.z().clone() - self.z().clone() * other.y().clone(),
            self.z().clone() * other.x().clone() - self.x().clone() * other.z().clone(),
            self.x().clone() * other.y().clone() - self.y().clone() * other.x().clone()
        )
    }
}

pub trait VectorXYZW<T> {
    fn x(&self) -> &T;
    fn x_mut(&mut self) -> &mut T;
    fn y(&self) -> &T;
    fn y_mut(&mut self) -> &mut T;
    fn z(&self) -> &T;
    fn z_mut(&mut self) -> &mut T;
    fn w(&self) -> &T;
    fn w_mut(&mut self) -> &mut T;
    fn xy(&self) -> Vector<T, 2>
        where T: Clone;
    fn yz(&self) -> Vector<T, 2>
        where T: Clone;
    fn zw(&self) -> Vector<T, 2>
        where T: Clone;
    fn xyz(&self) -> Vector<T, 3>
        where T: Clone;
    fn yzw(&self) -> Vector<T, 3>
        where T: Clone;
}

impl<T> VectorX<T> for Vector<T, 1> {
    fn x(&self) -> &T {
        &self.components[0]
    }
    fn x_mut(&mut self) -> &mut T {
        &mut self.components[0]
    }
}

impl<T> VectorXY<T> for Vector<T, 2> {
    fn x(&self) -> &T {
        &self.components[0]
    }
    fn x_mut(&mut self) -> &mut T {
        &mut self.components[0]
    }
    fn y(&self) -> &T {
        &self.components[1]
    }
    fn y_mut(&mut self) -> &mut T {
        &mut self.components[1]
    }
}

impl<T> VectorXYZ<T> for Vector<T, 3> {
    fn x(&self) -> &T {
        &self.components[0]
    }
    fn x_mut(&mut self) -> &mut T {
        &mut self.components[0]
    }
    fn y(&self) -> &T {
        &self.components[1]
    }
    fn y_mut(&mut self) -> &mut T {
        &mut self.components[1]
    }
    fn z(&self) -> &T {
        &self.components[2]
    }
    fn z_mut(&mut self) -> &mut T {
        &mut self.components[2]
    }
    fn xy(&self) -> Vector<T, 2>
        where T: Clone
    {
        vector!(self.x().clone(), self.y().clone())
    }
    fn yz(&self) -> Vector<T, 2>
        where T: Clone
    {
        vector!(self.y().clone(), self.z().clone())
    }
}

impl<T> VectorXYZW<T> for Vector<T, 4> {
    fn x(&self) -> &T {
        &self.components[0]
    }
    fn x_mut(&mut self) -> &mut T {
        &mut self.components[0]
    }
    fn y(&self) -> &T {
        &self.components[1]
    }
    fn y_mut(&mut self) -> &mut T {
        &mut self.components[1]
    }
    fn z(&self) -> &T {
        &self.components[2]
    }
    fn z_mut(&mut self) -> &mut T {
        &mut self.components[2]
    }
    fn w(&self) -> &T {
        &self.components[3]
    }
    fn w_mut(&mut self) -> &mut T {
        &mut self.components[3]
    }
    fn xy(&self) -> Vector<T, 2>
        where T: Clone
    {
        vector!(self.x().clone(), self.y().clone())
    }
    fn yz(&self) -> Vector<T, 2>
        where T: Clone
    {
        vector!(self.y().clone(), self.z().clone())
    }
    fn zw(&self) -> Vector<T, 2>
        where T: Clone
    {
        vector!(self.z().clone(), self.w().clone())
    }
    fn xyz(&self) -> Vector<T, 3>
        where T: Clone
    {
        vector!(self.x().clone(), self.y().clone(), self.z().clone())
    }
    fn yzw(&self) -> Vector<T, 3>
        where T: Clone
    {
        vector!(self.y().clone(), self.z().clone(), self.w().clone())
    }
}
