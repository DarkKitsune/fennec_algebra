/*pub enum ConstAssert<const COND: bool> {}

pub trait IsTrue {}
pub trait IsFalse {}

impl IsTrue for ConstAssert<true> {}
impl IsFalse for ConstAssert<false> {}*/

#[macro_export]
macro_rules! init_array {
    ([$value_type:ty; $count:expr], mut $value_fn:expr) => {{
        use std::mem::MaybeUninit;
        let mut func = $value_fn;
        let mut array: MaybeUninit<[$value_type; $count]> = MaybeUninit::uninit();
        unsafe {
            for idx in 0..$count {
                (array.as_mut_ptr() as *mut $value_type)
                    .add(idx)
                    .write(func(idx));
            }
            array.assume_init()
        }
    }};
    ([$value_type:ty; $count:expr], $value_fn:expr) => {{
        use std::mem::MaybeUninit;
        let func = $value_fn;
        let mut array: MaybeUninit<[$value_type; $count]> = MaybeUninit::uninit();
        unsafe {
            for idx in 0..$count {
                (array.as_mut_ptr() as *mut $value_type)
                    .add(idx)
                    .write(func(idx));
            }
            array.assume_init()
        }
    }};
}

#[macro_export]
macro_rules! count_args {
    ($($arg:expr),+$(,)?) => {
        [$(stringify!($arg)),+].len()
    }
}
