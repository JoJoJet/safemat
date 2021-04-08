pub trait Zero {
    fn zero() -> Self;
}
pub trait One {
    fn one() -> Self;
}

macro_rules! impl_zero_one {
    ($num: ty) => {
        impl $crate::Zero for $num {
            fn zero() -> Self {
                0
            }
        }
        impl $crate::One for $num {
            fn one() -> Self {
                1
            }
        }
    };
    ($($num: ty),*) => {
        $(impl_zero_one!{$num})*
    }
}
macro_rules! impl_zero_one_f {
    ($num: ty) => {
        impl $crate::Zero for $num {
            fn zero() -> Self {
                0.0
            }
        }
        impl $crate::One for $num {
            fn one() -> Self {
                1.0
            }
        }
    };
}

impl_zero_one!(usize, isize, i8, i16, i32, i64, i128, u8, u16, u32, u64, u128);
impl_zero_one_f!(f32);
impl_zero_one_f!(f64);
