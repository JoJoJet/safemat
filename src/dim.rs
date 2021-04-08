pub trait Dim: Copy {
    fn dim(&self) -> usize;
}

pub trait FixedDim: Dim {
    const DIM: usize;
    fn new() -> Self;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fixed<const N: usize>;

impl<const N: usize> Dim for Fixed<N> {
    #[inline]
    fn dim(&self) -> usize {
        N
    }
}

impl<const N: usize> FixedDim for Fixed<N> {
    const DIM: usize = N;
    #[inline]
    fn new() -> Self {
        Self
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plus<A: Dim, B: Dim>(pub A, pub B);

impl<A: Dim, B: Dim> Dim for Plus<A, B> {
    #[inline]
    fn dim(&self) -> usize {
        self.0.dim() + self.1.dim()
    }
}

impl<A: FixedDim, B: FixedDim> FixedDim for Plus<A, B> {
    const DIM: usize = A::DIM + B::DIM;
    #[inline]
    fn new() -> Self {
        Self(A::new(), B::new())
    }
}

macro_rules! impl_plus_into {
    ($sum: literal; $sub: literal) => {
        impl From<Plus::<Fixed::<{$sum - $sub}>, Fixed::<$sub>>> for Fixed::<$sum> {
            fn from(_: Plus::<Fixed::<{$sum - $sub}>, Fixed::<$sub>>) -> Self {
                Self
            }
        }
        impl From<Fixed::<$sum>> for Plus::<Fixed::<{$sum - $sub}>, Fixed::<$sub>> {
            fn from(_: Fixed::<$sum>) -> Self {
                Self(Fixed, Fixed)
            }
        }
    };
    ($sum: literal; $($sub: literal),*) => {
        $(impl_plus_into!($sum; $sub);)*
    };
}

impl_plus_into!(2; 1);
impl_plus_into!(3; 1, 2);
impl_plus_into!(4; 1, 2, 3);
impl_plus_into!(5; 1, 2, 3, 4);
impl_plus_into!(6; 1, 2, 3, 4, 5);
impl_plus_into!(7; 1, 2, 3, 4, 5, 6);
impl_plus_into!(8; 1, 2, 3, 4, 5, 6, 7);
impl_plus_into!(9; 1, 2, 3, 4, 5, 6, 7, 8);
impl_plus_into!(10; 1, 2, 3, 4, 5, 6, 7, 8, 9);
impl_plus_into!(11; 1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
impl_plus_into!(12; 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
impl_plus_into!(13; 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
impl_plus_into!(14; 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13);
impl_plus_into!(15; 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14);
impl_plus_into!(16; 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

/// Evaluates to an expression implementing [safemat::Dim], based on the input expression.
/// If the input is a usize literal, this will evaluate to a monomorphiztation of [safemat::FixedMat].
/// If the input is a variable, this will evaluate to a new, unique type, bound to that variable.
#[macro_export]
macro_rules! dim {
    ($val: literal) => {
        $crate::Fixed::<{ $val }>
    };

    ($var: ident) => {{
        mod $var {
            #[derive(Clone, Copy, Debug, PartialEq, Eq)]
            pub(super) struct VarDim(pub(super) usize);
            impl $crate::Dim for VarDim {
                fn dim(&self) -> usize {
                    self.0
                }
            }
        }
    }};
}
