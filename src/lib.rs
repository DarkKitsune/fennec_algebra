#![allow(incomplete_features)]
#![feature(const_generics)]
#![feature(const_evaluatable_checked)]

mod types;
pub use types::*;
mod traits;
pub use traits::*;
mod util;

#[cfg(test)]
mod tests {
    mod vector {
        use crate::vector;
        #[test]
        fn vector_new() {
            let vector = vector!(0, 44, 2);
            assert_eq!(*vector.component(0).unwrap(), 0);
            assert_eq!(*vector.component(2).unwrap(), 2);
            assert_eq!(*vector.component(1).unwrap(), 44);
            let vector = vector!(24.1, 4.44, 6.32, 666.420);
            assert_eq!(*vector.component(0).unwrap(), 24.1);
            assert_eq!(*vector.component(1).unwrap(), 4.44);
            assert_eq!(*vector.component(2).unwrap(), 6.32);
            assert_eq!(*vector.component(3).unwrap(), 666.420);
        }

        #[test]
        fn vector_length2() {
            let vector = vector!(2, 3, 4);
            assert_eq!(vector.length2().unwrap(), 29);
        }

        #[test]
        fn vector_length() {
            let vector = vector!(3.0f32, 4.0f32);
            assert_eq!(vector.length().unwrap(), 5.0f32);
        }

        #[test]
        fn vector_normalized() {
            let vector = vector!(3.0f32, 4.0f32);
            assert_eq!(
                vector.normalized().unwrap(),
                vector!(3.0f32 / 5.0f32, 4.0f32 / 5.0f32)
            );
        }

        #[test]
        fn vector_math() {
            // Addition
            let a = vector!(2, 3, 99);
            let b = vector!(6, -1, 2);
            assert_eq!(a + b, vector!(8, 2, 101));
            let mut a = a;
            a += vector!(2, 2, 1);
            assert_eq!(a, vector!(4, 5, 100));

            // Subtraction
            let a = vector!(2, 3, 99);
            let b = vector!(6, -1, 2);
            assert_eq!(a - b, vector!(-4, 4, 97));
            let mut a = a;
            a -= vector!(2, 2, 1);
            assert_eq!(a, vector!(0, 1, 98));

            // Multiplication
            let a = vector!(2, 3, 99);
            let b = vector!(6, -1, 2);
            assert_eq!(a * b, vector!(12, -3, 198));
            let mut a = a;
            a *= vector!(2, 2, 1);
            assert_eq!(a, vector!(4, 6, 99));

            // Division
            let a = vector!(2, 3, 99);
            let b = vector!(6, -1, 2);
            assert_eq!(a / b, vector!(0, -3, 49));
            let mut a = a;
            a /= vector!(2, 2, 1);
            assert_eq!(a, vector!(1, 1, 99));
        }
    }

    mod matrix {
        use crate::{vector, Matrix};
        #[test]
        fn matrix_tests() {
            // Create 4x4 matrix of 32-bit floats
            let mut ident = Matrix::<f32, 4, 4>::identity();

            // Test
            assert_eq!(ident[0], vector!(1.0, 0.0, 0.0, 0.0));
            assert_eq!(ident[1], vector!(0.0, 1.0, 0.0, 0.0));
            assert_eq!(ident[2], vector!(0.0, 0.0, 1.0, 0.0));
            assert_eq!(ident[3], vector!(0.0, 0.0, 0.0, 1.0));

            // Set the position part (column number 3)
            ident.set_position(vector!(1.0, 2.0, 3.0)).unwrap();

            // Test
            assert_eq!(ident[3], vector!(1.0, 2.0, 3.0, 1.0));

            // Multiply by a matrix with a position of (1.0, 0.0, 1.0) and a scale of (2.0, 2.0, 2.0)
            let multiplied = ident
                * Matrix::new_position_scale(vector!(1.0, 0.0, 1.0), vector!(2.0, 2.0, 2.0))
                    .unwrap();

            // Test
            assert_eq!(multiplied.position().unwrap(), vector!(3.0, 4.0, 7.0));
        }
    }

    mod nnet {
        use crate::NNet;

        #[test]
        fn nnet_new() {
            // Create network
            let mut seed = 13473; // Seed for random weights
            let mut nn: NNet<3, 1, 2, 2> = NNet::new(&mut seed, 0.02).unwrap();

            // Set up the training inputs and targets
            let inputs = [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
            ];
            let targets = [[0.0], [0.0], [0.0], [1.0], [0.0], [0.0], [1.0], [1.0]];

            // Train the network
            for _ in 0..10000 {
                for (input, targets) in inputs.iter().zip(targets.iter()) {
                    let _ = nn.forward(&input);
                    nn.backward(targets);
                }
            }

            // Test the network
            println!("{:?}", nn.forward(&[1.0, 1.0, 0.0]).unwrap());
            println!("{:?}", nn.forward(&[0.0, 0.0, 1.0]).unwrap());
            println!("{:?}", nn.forward(&[1.0, 0.0, 1.0]).unwrap());
            println!("{:?}", nn.forward(&[0.0, 1.0, 1.0]).unwrap());
            println!("{:?}", nn.forward(&[0.0, 0.0, 0.0]).unwrap());
        }

        #[test]
        fn nnet_train() {}
    }
}
