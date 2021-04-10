/// A value representing the length of a dimension of a matrix.
pub trait Dim: Copy {
    fn dim(&self) -> usize;
}

/// A matrix dimension whose length is known at compile-time.
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

/// The sum of two dimensions.
/// You will usually see this type as the result of matrix concatenation.
/// If both operands are constants, you can combine them with the
/// [`Patch`] trait's `patch` method.
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

/// A composite dimension (e.g. [`Plus`]) that can be reduced to a single dimension.
pub trait Patch: Dim {
    type Target: Dim;
    fn patch(self) -> Self::Target;
}

macro_rules! impl_plus_into {
    ($sum: literal;; $sub: expr) => {
        impl Patch for Plus::<Fixed::<{$sum - $sub}>, Fixed::<{$sub}>> {
            type Target = Fixed::<{$sum}>;
            fn patch(self) -> Self::Target {
                Fixed
            }
        }
        impl From<Plus::<Fixed::<{$sum - $sub}>, Fixed::<{$sub}>>> for Fixed::<{$sum}> {
            fn from(_: Plus::<Fixed::<{$sum - $sub}>, Fixed::<{$sub}>>) -> Self {
                Self
            }
        }
        impl From<Fixed::<$sum>> for Plus::<Fixed::<{$sum - $sub}>, Fixed::<{$sub}>> {
            fn from(_: Fixed::<$sum>) -> Self {
                Self(Fixed, Fixed)
            }
        }
    };
    ($sum: literal; $($sub: literal),* ;) => {
        $(impl_plus_into!($sum;; $sub);)*
        $(impl_plus_into!($sum;; { $sum - $sub });)*
        impl_plus_into!($sum;; {$sum / 2});
    };
    ($sum: literal; $($sub: literal),*) => {
        $(impl_plus_into!($sum;; $sub);)*
        $(impl_plus_into!($sum;; { $sum - $sub});)*
    };
}

impl_plus_into!(2; 0 ;);
impl_plus_into!(3; 0, 1);
impl_plus_into!(4; 0, 1 ;);
impl_plus_into!(5; 0, 1, 2);
impl_plus_into!(6; 0, 1, 2 ;);
impl_plus_into!(7; 0, 1, 2, 3);
impl_plus_into!(8; 0, 1, 2, 3 ;);
impl_plus_into!(9; 0, 1, 2, 3, 4);
impl_plus_into!(10; 0, 1, 2, 3, 4 ;);
impl_plus_into!(11; 0, 1, 2, 3, 4, 5);
impl_plus_into!(12; 0, 1, 2, 3, 4, 5 ;);
impl_plus_into!(13; 0, 1, 2, 3, 4, 5, 6);
impl_plus_into!(14; 0, 1, 2, 3, 4, 5, 6 ;);
impl_plus_into!(15; 0, 1, 2, 3, 4, 5, 6, 7);
impl_plus_into!(16; 0, 1, 2, 3, 4, 5, 6, 7 ;);
impl_plus_into!(32; 0, 8 ;);
impl_plus_into!(64; 0, 16 ;);
impl_plus_into!(128; 0, 32 ;);
impl_plus_into!(256; 0, 64 ;);
impl_plus_into!(512; 0, 128 ;);
impl_plus_into!(1024; 0, 256 ;);
impl_plus_into!(2048; 0, 512 ;);
impl_plus_into!(4096; 0, 1024 ;);
impl_plus_into!(8192; 0, 2048 ;);
impl_plus_into!(16_384; 0, 4096 ;);
impl_plus_into!(32_768; 0, 8192 ;);

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
