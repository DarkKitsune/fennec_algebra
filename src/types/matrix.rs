use crate::*;
use std::fmt::{Debug, Formatter};
use std::iter::Sum;
use std::ops::{Add, Div, Index, IndexMut, Mul, Rem, Sub};

#[repr(C)]
pub struct Matrix<T: Sized, const COLUMNS: usize, const ROWS: usize> {
    columns: Vector<Vector<T, ROWS>, COLUMNS>,
}

impl<T: Sized, const COLUMNS: usize, const ROWS: usize> Matrix<T, COLUMNS, ROWS> {
    pub const fn new(columns: Vector<Vector<T, ROWS>, COLUMNS>) -> Self {
        Self { columns }
    }

    pub const fn from_array(columns: [Vector<T, ROWS>; COLUMNS]) -> Self {
        let columns = Vector::new(columns);
        Self::new(columns)
    }

    pub fn from_array_array(columns: &[[T; ROWS]; COLUMNS]) -> Self
    where
        T: Clone,
    {
        Self::from_array(init_array!([Vector<T, ROWS>; COLUMNS], |idx| {
            let column: &[T; ROWS] = &columns[idx];
            Vector::new(column.clone())
        }))
    }

    pub fn from_smaller_array_array<const R: usize, const C: usize>(
        columns: &[[T; R]; C],
    ) -> Result<Self, MatrixError>
    where
        T: Clone + Zero + One,
    {
        if R > ROWS {
            return Err(MatrixError::TooFewRows);
        }
        if C > COLUMNS {
            return Err(MatrixError::TooFewColumns);
        }
        Ok(Self::from_array(init_array!(
            [Vector<T, ROWS>; COLUMNS],
            |idx| {
                if idx < C {
                    let column: &[T; R] = &columns[idx];
                    let new_column = init_array!([T; ROWS], |r_idx| if r_idx < R {
                        let v: &T = &column[r_idx];
                        v.clone()
                    } else {
                        if r_idx == idx {
                            T::one()
                        } else {
                            T::zero()
                        }
                    });
                    Vector::new(new_column)
                } else {
                    Vector::new(init_array!([T; ROWS], |r_idx| if r_idx == idx {
                        T::one()
                    } else {
                        T::zero()
                    }))
                }
            }
        )))
    }

    pub fn new_scale(scale: Vector<T, 3>) -> Result<Self, MatrixError>
    where
        T: Add<T, Output = T> + Sqrt<Output = T> + Clone,
        T: Mul<T, Output = T>,
        T: Div<T, Output = T>,
        Vector<T, ROWS>: Mul<T, Output = Vector<T, ROWS>>,
        T: Default + One + Clone,
    {
        let mut mat = Self::identity();
        mat.set_scale(scale)?;
        Ok(mat)
    }

    pub fn new_position(position: Vector<T, 3>) -> Result<Self, MatrixError>
    where
        T: Default + One + Clone,
    {
        let mut mat = Self::identity();
        mat.set_position(position)?;
        Ok(mat)
    }

    pub fn new_position_scale(
        position: Vector<T, 3>,
        scale: Vector<T, 3>,
    ) -> Result<Self, MatrixError>
    where
        T: Add<T, Output = T> + Sqrt<Output = T> + Clone,
        T: Mul<T, Output = T>,
        T: Div<T, Output = T>,
        Vector<T, ROWS>: Mul<T, Output = Vector<T, ROWS>>,
        T: Default + One + Clone,
    {
        let mut mat = Self::identity();
        mat.set_position(position)?;
        mat.set_scale(scale)?;
        Ok(mat)
    }

    pub fn row_length(&self) -> usize {
        COLUMNS
    }

    pub fn row_count(&self) -> usize {
        ROWS
    }

    pub fn column_length(&self) -> usize {
        ROWS
    }

    pub fn column_count(&self) -> usize {
        COLUMNS
    }

    pub fn column(&self, idx: usize) -> &Vector<T, ROWS> {
        &self.columns[idx]
    }

    pub fn column_mut(&mut self, idx: usize) -> &mut Vector<T, ROWS> {
        &mut self.columns[idx]
    }

    pub fn row(&self, idx: usize) -> Vector<T, COLUMNS>
    where
        T: Clone,
    {
        Vector::new(init_array!([T; COLUMNS], |column_idx| self.columns
            [column_idx][idx]
            .clone()))
    }

    fn _row_with_column_size(&self, idx: usize) -> Vector<T, ROWS>
    where
        T: Clone,
    {
        Vector::new(init_array!([T; ROWS], |column_idx| self.columns
            [column_idx][idx]
            .clone()))
    }

    pub fn identity() -> Self
    where
        T: Default + One,
    {
        Self::new(Vector::new(init_array!(
            [Vector<T, ROWS>; COLUMNS],
            |column_idx| {
                Vector::new(init_array!([T; ROWS], |row_idx| if row_idx == column_idx {
                    T::one()
                } else {
                    T::default()
                }))
            }
        )))
    }

    pub fn add_matrix(&mut self, other: &Self)
    where
        T: Add<T, Output = T> + Clone,
    {
        self.columns += &other.columns;
    }

    pub fn sub_matrix(&mut self, other: &Self)
    where
        T: Sub<T, Output = T> + Clone,
    {
        self.columns -= &other.columns;
    }

    pub fn mul_matrix(&self, other: &Matrix<T, ROWS, COLUMNS>) -> Matrix<T, ROWS, COLUMNS>
    where
        T: Mul<T, Output = T> + Clone + Sum,
    {
        let columns = Vector::new(init_array!([Vector<T, COLUMNS>; ROWS], |row_idx| {
            Vector::new(init_array!([T; COLUMNS], |column_idx| {
                let column = other.row(column_idx);
                self.column(row_idx).dot(&column)
            }))
        }));
        Matrix::new(columns)
    }

    pub fn div_matrix(&mut self, other: &Self)
    where
        T: Div<T, Output = T> + Clone,
    {
        self.columns /= &other.columns;
    }

    pub fn rem_matrix(&mut self, other: &Self)
    where
        T: Rem<T, Output = T> + Clone,
    {
        self.columns %= &other.columns;
    }

    pub fn set_position(&mut self, position: Vector<T, 3>) -> Result<(), MatrixError>
    where
        T: Clone,
    {
        if ROWS < 3 {
            return Err(MatrixError::TooFewRows);
        }
        let column = self.column_mut(COLUMNS - 1);
        column[0] = position[0].clone();
        column[1] = position[1].clone();
        column[2] = position[2].clone();
        Ok(())
    }

    pub fn position(&self) -> Result<Vector<T, 3>, MatrixError>
    where
        T: Clone,
    {
        if ROWS < 3 {
            return Err(MatrixError::TooFewRows);
        }
        let column = &self.columns[COLUMNS - 1];
        Ok(vector!(
            column[0].clone(),
            column[1].clone(),
            column[2].clone()
        ))
    }

    pub fn set_scale(&mut self, scale: Vector<T, 3>) -> Result<(), MatrixError>
    where
        T: Add<T, Output = T> + Sqrt<Output = T> + Clone,
        T: Mul<T, Output = T>,
        T: Div<T, Output = T>,
        Vector<T, ROWS>: Mul<T, Output = Vector<T, ROWS>>,
    {
        if ROWS < 3 {
            return Err(MatrixError::TooFewRows);
        }
        if COLUMNS < 3 {
            return Err(MatrixError::TooFewColumns);
        }
        self.columns[0] = self.columns[0]
            .normalized()
            .map_err(|err| MatrixError::VectorError(err))?
            * scale[0].clone();
        self.columns[1] = self.columns[1]
            .normalized()
            .map_err(|err| MatrixError::VectorError(err))?
            * scale[1].clone();
        self.columns[2] = self.columns[2]
            .normalized()
            .map_err(|err| MatrixError::VectorError(err))?
            * scale[2].clone();
        Ok(())
    }

    pub fn scale(&self) -> Result<Vector<T, 3>, MatrixError>
    where
        T: Add<T, Output = T> + Mul<T, Output = T> + Sqrt<Output = T> + Clone,
    {
        if ROWS < 3 {
            return Err(MatrixError::TooFewRows);
        }
        if COLUMNS < 3 {
            return Err(MatrixError::TooFewColumns);
        }
        Ok(vector!(
            self.columns[0]
                .length()
                .map_err(|err| MatrixError::VectorError(err))?,
            self.columns[1]
                .length()
                .map_err(|err| MatrixError::VectorError(err))?,
            self.columns[2]
                .length()
                .map_err(|err| MatrixError::VectorError(err))?,
        ))
    }

    pub fn set_x(&mut self, x: Vector<T, 3>) -> Result<(), MatrixError>
    where
        T: Clone,
    {
        if ROWS < 3 {
            return Err(MatrixError::TooFewRows);
        }
        let column = self.column_mut(0);
        column[0] = x[0].clone();
        column[1] = x[1].clone();
        column[2] = x[2].clone();
        Ok(())
    }

    pub fn x(&self) -> Result<Vector<T, 3>, MatrixError>
    where
        T: Clone,
    {
        if ROWS < 3 {
            return Err(MatrixError::TooFewRows);
        }
        let column = &self.columns[0];
        Ok(vector!(
            column[0].clone(),
            column[1].clone(),
            column[2].clone()
        ))
    }

    pub fn set_y(&mut self, y: Vector<T, 3>) -> Result<(), MatrixError>
    where
        T: Clone,
    {
        if ROWS < 3 {
            return Err(MatrixError::TooFewRows);
        }
        let column = self.column_mut(1);
        column[0] = y[0].clone();
        column[1] = y[1].clone();
        column[2] = y[2].clone();
        Ok(())
    }

    pub fn y(&self) -> Result<Vector<T, 3>, MatrixError>
    where
        T: Clone,
    {
        if ROWS < 3 {
            return Err(MatrixError::TooFewRows);
        }
        let column = &self.columns[1];
        Ok(vector!(
            column[0].clone(),
            column[1].clone(),
            column[2].clone()
        ))
    }

    pub fn set_z(&mut self, z: Vector<T, 3>) -> Result<(), MatrixError>
    where
        T: Clone,
    {
        if ROWS < 3 {
            return Err(MatrixError::TooFewRows);
        }
        let column = self.column_mut(2);
        column[0] = z[0].clone();
        column[1] = z[1].clone();
        column[2] = z[2].clone();
        Ok(())
    }

    pub fn z(&self) -> Result<Vector<T, 3>, MatrixError>
    where
        T: Clone,
    {
        if ROWS < 3 {
            return Err(MatrixError::TooFewRows);
        }
        let column = &self.columns[2];
        Ok(vector!(
            column[0].clone(),
            column[1].clone(),
            column[2].clone()
        ))
    }

    pub fn transform_point(&self, point: Vector<T, 3>) -> Result<Vector<T, 3>, MatrixError>
    where
        T: Mul<T, Output = T> + Clone + Sum + Default + One,
    {
        let mut a = self.clone();
        let b = Matrix::new_position(point)?;
        Ok(a.mul_matrix(&b).position()?)
    }

    pub fn transposed(&self) -> Result<Self, MatrixError>
    where
        T: Clone,
    {
        if ROWS < COLUMNS {
            return Err(MatrixError::TooFewRows);
        }
        if COLUMNS < ROWS {
            return Err(MatrixError::TooFewColumns);
        }
        Ok(Self::from_array_array(&init_array!(
            [[T; ROWS]; COLUMNS],
            |c_idx| init_array!([T; ROWS], |r_idx| self.columns[r_idx][c_idx].clone())
        )))
    }

    pub fn convert<T2: Sized>(&self) -> Matrix<T2, COLUMNS, ROWS>
    where
        T: Into<T2> + Clone,
    {
        Matrix::new(Vector::new(init_array!(
            [Vector<T2, ROWS>; COLUMNS],
            |column_idx| Vector::new(init_array!([T2; ROWS], |row_idx| self[column_idx][row_idx]
                .clone()
                .into()))
        )))
    }
}

impl<T: Sized, const COLUMNS: usize, const ROWS: usize> PartialEq for Matrix<T, COLUMNS, ROWS>
where
    Vector<Vector<T, ROWS>, COLUMNS>: PartialEq,
{
    fn eq(&self, rhs: &Self) -> bool {
        self.columns == rhs.columns
    }
}

impl<T: Sized, const COLUMNS: usize, const ROWS: usize> Eq for Matrix<T, COLUMNS, ROWS> where
    Vector<Vector<T, ROWS>, COLUMNS>: Eq
{
}

impl<T: Sized, const COLUMNS: usize, const ROWS: usize> Clone for Matrix<T, COLUMNS, ROWS>
where
    Vector<Vector<T, ROWS>, COLUMNS>: Clone,
{
    fn clone(&self) -> Self {
        Self {
            columns: self.columns.clone(),
        }
    }
}

impl<T: Sized, const COLUMNS: usize, const ROWS: usize> Copy for Matrix<T, COLUMNS, ROWS> where
    Vector<Vector<T, ROWS>, COLUMNS>: Copy
{
}

impl<T: Sized, const COLUMNS: usize, const ROWS: usize> Debug for Matrix<T, COLUMNS, ROWS>
where
    Vector<Vector<T, ROWS>, COLUMNS>: Debug,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Matrix")
            .field("columns", &self.columns)
            .finish()
    }
}

impl<T: Sized, const COLUMNS: usize, const ROWS: usize> Index<usize> for Matrix<T, COLUMNS, ROWS> {
    type Output = Vector<T, ROWS>;

    fn index(&self, idx: usize) -> &Vector<T, ROWS> {
        &self.columns[idx]
    }
}

impl<T: Sized, const COLUMNS: usize, const ROWS: usize> IndexMut<usize>
    for Matrix<T, COLUMNS, ROWS>
{
    fn index_mut(&mut self, idx: usize) -> &mut Vector<T, ROWS> {
        &mut self.columns[idx]
    }
}

macro_rules! matrix_binary_op_mul {
    ($op:ident, $fn_name:ident, $type_method:ident) => {
        impl<'a, 'b, T: Sized, const COLUMNS: usize, const ROWS: usize>
            $op<&'b Matrix<T, ROWS, COLUMNS>> for &'a Matrix<T, COLUMNS, ROWS>
        where
            T: $op<T, Output = T> + Clone + Sum,
        {
            type Output = Matrix<T, ROWS, COLUMNS>;

            fn $fn_name(self, rhs: &'b Matrix<T, ROWS, COLUMNS>) -> Matrix<T, ROWS, COLUMNS> {
                self.$type_method(rhs)
            }
        }

        impl<'a, T: Sized, const COLUMNS: usize, const ROWS: usize>
            $op<&'a Matrix<T, ROWS, COLUMNS>> for Matrix<T, COLUMNS, ROWS>
        where
            T: $op<T, Output = T> + Clone + Sum,
        {
            type Output = Matrix<T, ROWS, COLUMNS>;

            fn $fn_name(self, rhs: &'a Matrix<T, ROWS, COLUMNS>) -> Matrix<T, ROWS, COLUMNS> {
                self.$type_method(rhs)
            }
        }

        impl<'a, T: Sized, const COLUMNS: usize, const ROWS: usize> $op<Matrix<T, ROWS, COLUMNS>>
            for &'a Matrix<T, COLUMNS, ROWS>
        where
            T: $op<T, Output = T> + Clone + Sum,
        {
            type Output = Matrix<T, ROWS, COLUMNS>;

            fn $fn_name(self, rhs: Matrix<T, ROWS, COLUMNS>) -> Matrix<T, ROWS, COLUMNS> {
                self.$type_method(&rhs)
            }
        }

        impl<T: Sized, const COLUMNS: usize, const ROWS: usize> $op<Matrix<T, ROWS, COLUMNS>>
            for Matrix<T, COLUMNS, ROWS>
        where
            T: $op<T, Output = T> + Clone + Sum,
        {
            type Output = Matrix<T, ROWS, COLUMNS>;

            fn $fn_name(self, rhs: Matrix<T, ROWS, COLUMNS>) -> Matrix<T, ROWS, COLUMNS> {
                self.$type_method(&rhs)
            }
        }
    };
}

macro_rules! matrix_binary_op {
    ($op:ident, $fn_name:ident, $type_method:ident) => {
        impl<'a, 'b, T: Sized, const COLUMNS: usize, const ROWS: usize>
            $op<&'b Matrix<T, COLUMNS, ROWS>> for &'a Matrix<T, COLUMNS, ROWS>
        where
            T: $op<T, Output = T> + Clone,
            Matrix<T, COLUMNS, ROWS>: Clone,
        {
            type Output = Matrix<T, COLUMNS, ROWS>;

            fn $fn_name(self, rhs: &'b Matrix<T, COLUMNS, ROWS>) -> Matrix<T, COLUMNS, ROWS> {
                let mut new = self.clone();
                new.$type_method(rhs);
                new
            }
        }

        impl<'a, T: Sized, const COLUMNS: usize, const ROWS: usize>
            $op<&'a Matrix<T, COLUMNS, ROWS>> for Matrix<T, COLUMNS, ROWS>
        where
            T: $op<T, Output = T> + Clone,
            Matrix<T, COLUMNS, ROWS>: Clone,
        {
            type Output = Matrix<T, COLUMNS, ROWS>;

            fn $fn_name(self, rhs: &'a Matrix<T, COLUMNS, ROWS>) -> Matrix<T, COLUMNS, ROWS> {
                let mut new = self.clone();
                new.$type_method(rhs);
                new
            }
        }

        impl<'a, T: Sized, const COLUMNS: usize, const ROWS: usize> $op<Matrix<T, COLUMNS, ROWS>>
            for &'a Matrix<T, COLUMNS, ROWS>
        where
            T: $op<T, Output = T> + Clone,
            Matrix<T, COLUMNS, ROWS>: Clone,
        {
            type Output = Matrix<T, COLUMNS, ROWS>;

            fn $fn_name(self, rhs: Matrix<T, COLUMNS, ROWS>) -> Matrix<T, COLUMNS, ROWS> {
                let mut new = self.clone();
                new.$type_method(&rhs);
                new
            }
        }

        impl<T: Sized, const COLUMNS: usize, const ROWS: usize> $op<Matrix<T, COLUMNS, ROWS>>
            for Matrix<T, COLUMNS, ROWS>
        where
            T: $op<T, Output = T> + Clone,
            Matrix<T, COLUMNS, ROWS>: Clone,
        {
            type Output = Matrix<T, COLUMNS, ROWS>;

            fn $fn_name(self, rhs: Matrix<T, COLUMNS, ROWS>) -> Matrix<T, COLUMNS, ROWS> {
                let mut new = self.clone();
                new.$type_method(&rhs);
                new
            }
        }
    };
}

matrix_binary_op!(Add, add, add_matrix);
matrix_binary_op!(Sub, sub, sub_matrix);
matrix_binary_op_mul!(Mul, mul, mul_matrix);
matrix_binary_op!(Div, div, div_matrix);
matrix_binary_op!(Rem, rem, rem_matrix);

pub trait TransformMatrix<T>: Sized {
    fn new_rotation(quat: &Quaternion) -> Result<Self, MatrixError>;
    fn new_rotation_on_axis(axis: Vector<T, 3>, radians: T) -> Result<Self, MatrixError>;
    fn view(from: Vector<T, 3>, to: Vector<T, 3>, up: Vector<T, 3>) -> Result<Self, MatrixError>;
    fn ortho(size: Vector<T, 2>, near: T, far: T) -> Self;
    fn projection(fov: T, aspect: T, near_plane: T, far_plane: T) -> Result<Self, MatrixError>;
}

impl TransformMatrix<f32> for Matrix<f32, 4, 4> {
    fn new_rotation(quat: &Quaternion) -> Result<Self, MatrixError> {
        let (axis, angle) = quat.axis_angle();
        Self::new_rotation_on_axis(axis, angle)
    }

    fn new_rotation_on_axis(axis: Vector<f32, 3>, radians: f32) -> Result<Self, MatrixError> {
        let axis = axis
            .normalized()
            .map_err(|err| MatrixError::VectorError(err))?;
        let (sin, cos) = (-radians).sin_cos();
        let t = 1.0 - cos;
        Ok(Matrix::new(vector!(
            vector!(
                t * axis[0] * axis[0] + cos,
                t * axis[0] * axis[1] - sin * axis[2],
                t * axis[0] * axis[2] + sin * axis[1],
                0.0,
            ),
            vector!(
                t * axis[0] * axis[1] + sin * axis[2],
                t * axis[1] * axis[1] + cos,
                t * axis[1] * axis[2] - sin * axis[0],
                0.0,
            ),
            vector!(
                t * axis[0] * axis[2] - sin * axis[1],
                t * axis[1] * axis[2] + sin * axis[0],
                t * axis[2] * axis[2] + cos,
                0.0,
            ),
            vector!(0.0, 0.0, 0.0, 1.0),
        )))
    }

    fn view(
        from: Vector<f32, 3>,
        to: Vector<f32, 3>,
        up: Vector<f32, 3>,
    ) -> Result<Self, MatrixError> {
        let zaxis = (from - to)
            .normalized()
            .map_err(|err| MatrixError::VectorError(err))?;
        let xaxis = up
            .cross(&zaxis)
            .normalized()
            .map_err(|err| MatrixError::VectorError(err))?;
        let yaxis = zaxis.cross(&xaxis);

        Ok(Self::new(vector!(
            vector!(xaxis[0], yaxis[0], zaxis[0], 0.0),
            vector!(xaxis[1], yaxis[1], zaxis[1], 0.0),
            vector!(xaxis[2], yaxis[2], zaxis[2], 0.0),
            vector!(-xaxis.dot(&from), -yaxis.dot(&from), -zaxis.dot(&from), 1.0),
        )))
    }

    fn ortho(size: Vector<f32, 2>, near: f32, far: f32) -> Self {
        Self::new(vector!(
            vector!(2.0 / size[0], 0.0, 0.0, 0.0),
            vector!(0.0, 2.0 / size[1], 0.0, 0.0),
            vector!(0.0, 0.0, 1.0 / (near - far), 0.0),
            vector!(0.0, 0.0, near / (near - far), 1.0),
        ))
    }

    fn projection(
        fov: f32,
        aspect: f32,
        near_plane: f32,
        far_plane: f32,
    ) -> Result<Self, MatrixError> {
        if fov <= 0.0 || fov >= std::f32::consts::PI {
            return Err(MatrixError::OutOfRangeFOV);
        }
        if near_plane <= 0.0 || near_plane >= far_plane {
            return Err(MatrixError::IncorrectNearFarPlanes);
        }

        let y_scale = 1.0 / (fov * 0.5).tan();
        let x_scale = y_scale / aspect;

        Ok(Self::new(vector!(
            vector!(x_scale, 0.0, 0.0, 0.0),
            vector!(0.0, y_scale, 0.0, 0.0),
            vector!(0.0, 0.0, far_plane / (near_plane - far_plane), -1.0),
            vector!(
                0.0,
                0.0,
                near_plane * far_plane / (near_plane - far_plane),
                0.0,
            ),
        )))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Hash)]
pub enum MatrixError {
    NotSquare,
    TooFewRows,
    TooFewColumns,
    OutOfRangeFOV,
    IncorrectNearFarPlanes,
    VectorError(VectorError),
}
