use std::ops::Add;

/// A value representing the length of a dimension of a matrix.
pub trait Dim: Copy {
    fn dim(&self) -> usize;
}

/// A composite dimension (e.g. [`Plus`]) that can be reduced to a single dimension.
pub trait Patch: Dim {
    type Target: Dim;
    fn patch(self) -> Self::Target;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fixed<const N: usize>;

impl<const N: usize> Dim for Fixed<N> {
    #[inline]
    fn dim(&self) -> usize {
        N
    }
}

impl<const N: usize> Patch for Fixed<N> {
    type Target = Self;
    #[inline]
    fn patch(self) -> Self {
        self
    }
}

/// The sum of two dimensions.
#[derive(Clone, Copy, Debug, Eq)]
pub struct Plus<A: Dim, B: Dim>(pub A, pub B);

impl<A: Dim, B: Dim> Dim for Plus<A, B> {
    #[inline]
    fn dim(&self) -> usize {
        self.0.dim() + self.1.dim()
    }
}

impl<A: Dim, B: Dim, C: Dim> PartialEq<C> for Plus<A, B> {
    #[inline]
    fn eq(&self, rhs: &C) -> bool {
        self.dim() == rhs.dim()
    }
}

impl<A: Dim, B: Dim> Patch for Plus<A, B> {
    type Target = Self;
    #[inline]
    fn patch(self) -> Self {
        self
    }
}


macro_rules! impl_add {
    ($sum: literal;; $sub: expr) => {
        impl Add<Fixed::<{$sub}>> for Fixed::<{$sum - $sub}> {
            type Output = Fixed::<{$sum}>;
            #[inline]
            fn add(self, _: Fixed::<{$sub}>) -> Self::Output {
                Fixed
            }
        }
    };
    ($sum: literal; $($sub: literal),* ;) => {
        $(impl_add!($sum;; $sub);)*
        $(impl_add!($sum;; { $sum - $sub });)*
        impl_add!($sum;; {$sum / 2});
    };
    ($sum: literal; $($sub: literal),*) => {
        $(impl_add!($sum;; $sub);)*
        $(impl_add!($sum;; { $sum - $sub});)*
    };
}

impl_add!(2; 0 ;);
impl_add!(3; 0, 1);
impl_add!(4; 0, 1 ;);
impl_add!(5; 0, 1, 2);
impl_add!(6; 0, 1, 2 ;);
impl_add!(7; 0, 1, 2, 3);
impl_add!(8; 0, 1, 2, 3 ;);
impl_add!(9; 0, 1, 2, 3, 4);
impl_add!(10; 0, 1, 2, 3, 4 ;);
impl_add!(11; 0, 1, 2, 3, 4, 5);
impl_add!(12; 0, 1, 2, 3, 4, 5 ;);
impl_add!(13; 0, 1, 2, 3, 4, 5, 6);
impl_add!(14; 0, 1, 2, 3, 4, 5, 6 ;);
impl_add!(15; 0, 1, 2, 3, 4, 5, 6, 7);
impl_add!(16; 0, 1, 2, 3, 4, 5, 6, 7 ;);
impl_add!(32; 0, 8 ;);
impl_add!(64; 0, 16 ;);
impl_add!(128; 0, 32 ;);
impl_add!(256; 0, 64 ;);
impl_add!(512; 0, 128 ;);
impl_add!(1024; 0, 256 ;);
impl_add!(2048; 0, 512 ;);
impl_add!(4096; 0, 1024 ;);
impl_add!(8192; 0, 2048 ;);
impl_add!(16_384; 0, 4096 ;);
impl_add!(32_768; 0, 8192 ;);

/// Evaluates to an expression implementing [safemat::Dim], based on the input expression.
/// If the input is a usize literal, this will evaluate to a monomorphiztation of [safemat::FixedMat].
/// If the input is a variable, this will evaluate to a new, unique type, bound to that variable.
///
/// This allows you to have matrices whose dimensions are not known at compile time without abandoning type-safety.
/// In order to perform operations such as multiplication or concatenation, the matrix dimensions must agree, so
/// usually they must be known at compile time.
/// However, if the dimensions of two matrices both depend on the *same* variable, we know that they must agree,
/// even if their specific values are not known at compile time.
/// ```
/// # use safemat::*;
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// #   do_patch(Some("4".to_owned()).into_iter())
/// # }
/// # fn do_patch(mut args: impl Iterator<Item = String>) -> Result<(), Box<dyn std::error::Error>> {
/// // Grab an unkown number from the command line.
/// let len = args.next()
///     .ok_or("no args")?
///     .parse::<usize>()?;
/// // Build a matrix, using the number as a dimension.
/// let a = Matrix::from_fn_with_dim(dim!(len), dim!(2), |i, j| i + j);
/// # assert_eq!(a, mat![ 0, 1 ; 1, 2 ; 2, 3 ; 3, 4 ]);
/// let b = mat![ 1, 2, 3 ; 4, 5, 6 ];
/// let c = &a * &b; // Since a.n == b.m, we can multiply them.
///                  // `c` has dimensions `len` x 3.
/// # assert_eq!(c, mat![ 4, 5, 6 ; 9, 12, 15 ; 14, 19, 24 ; 19, 26, 33 ]);
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! dim {
    ($val: literal) => {
        $crate::Fixed::<{ $val }>
    };

    ($var: ident) => {{
        mod var_dim {
            #[allow(non_camel_case_types)]
            #[derive(Clone, Copy, Debug, Eq)]
            pub(super) struct $var(pub(super) usize);
            impl $crate::Dim for $var {
                #[inline]
                fn dim(&self) -> usize {
                    self.0
                }
            }

            use $crate::Dim as _;
            impl<B: $crate::Dim> std::cmp::PartialEq<B> for $var {
                #[inline]
                fn eq(&self, rhs: &B) -> bool {
                    self.0 == rhs.dim()
                }
            }
            impl<const N: usize> std::cmp::PartialEq<$var> for $crate::Fixed<N> {
                #[inline]
                fn eq(&self, v: &$var) -> bool {
                    N == v.0
                }
            }

            impl<B: $crate::Dim> std::ops::Add<B> for $var {
                type Output = $crate::Plus<Self, B>;
                #[inline]
                fn add(self, rhs: B) -> Self::Output {
                    $crate::Plus(self, rhs)
                }
            }
            impl<const N: usize> std::ops::Add<$var> for $crate::Fixed<N> {
                type Output = $crate::Plus<$crate::Fixed<N>, $var>;
                #[inline]
                fn add(self, rhs: $var) -> Self::Output {
                    $crate::Plus(self, rhs)
                }
            }

            impl $crate::Patch for $var {
                type Target = Self;
                #[inline]
                fn patch(self) -> Self {
                    self
                }
            }
        }
        var_dim::$var($var)
    }};
}
