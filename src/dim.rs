use std::{marker::PhantomData, ops::Add};

/// A value representing the length of a dimension of a matrix.
pub trait Dim: Copy + 'static {
    fn dim(&self) -> usize;
}

/// A marker trait which asserts that the implementing type is just
/// a different way of expressing the dimension `B`.
///
/// For example: [`Plus<A, B>`] = [`Plus<B, A>`].
/// ```
/// # use safemat::*;
/// let len = 2;         // some spoopy unknown number.
/// let len = dim!(len); // store the dim in a variable since the type is unnameable.
/// let a = Matrix::from_fn_with_dim(dim!(1), len, |_, j| j + 1);
/// let a = a.hcat(mat![3]); // N: Plus<len, 1>
///
/// let b = Matrix::from_fn_with_dim(len, dim!(1), |i, _| i + 2);
/// let b = mat![1].vcat(b); // M: Plus<1, len>.
///
/// let c = &a * &b; // We can multiply them, because the Identity trait
///                  // tells the type system that Plus<len, 1> = Plus<1, len>.
/// assert_eq!(c, mat![14]);
/// ```
pub trait Identity<B: Dim>: Dim {
    /// Converts this instance to the target type (should be zero-cost).
    fn identity(self) -> B;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fixed<const N: usize>;

impl<const N: usize> Dim for Fixed<N> {
    #[inline]
    fn dim(&self) -> usize {
        N
    }
}

impl<const N: usize> Identity<Fixed<N>> for Fixed<N> {
    #[inline]
    fn identity(self) -> Self {
        self
    }
}

/// The sum of two dimensions.
#[derive(Clone, Copy, Debug, Eq)]
pub struct Plus<A: Dim, B: Dim> {
    val: usize,
    _a: PhantomData<A>,
    _b: PhantomData<B>,
}

impl<A: Dim, B: Dim> Plus<A, B> {
    #[inline]
    pub fn new(a: A, b: B) -> Self {
        Self {
            val: a.dim() + b.dim(),
            _a: PhantomData,
            _b: PhantomData,
        }
    }
}

impl<A: Dim, B: Dim> Dim for Plus<A, B> {
    #[inline]
    fn dim(&self) -> usize {
        self.val
    }
}

impl<A: Dim, B: Dim, C: Dim> PartialEq<C> for Plus<A, B> {
    #[inline]
    fn eq(&self, rhs: &C) -> bool {
        self.val == rhs.dim()
    }
}

impl<A: Dim, B: Dim> Identity<Plus<B, A>> for Plus<A, B> {
    #[inline]
    fn identity(self) -> Plus<B, A> {
        Plus {
            val: self.val,
            _a: PhantomData,
            _b: PhantomData,
        }
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
        $crate::dim::Fixed::<{ $val }>
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

            impl $crate::dim::Identity<$var> for $var {
                #[inline]
                fn identity(self) -> Self {
                    self
                }
            }

            use $crate::Dim as _;
            impl<B: $crate::Dim> std::cmp::PartialEq<B> for $var {
                #[inline]
                fn eq(&self, rhs: &B) -> bool {
                    self.0 == rhs.dim()
                }
            }
            impl<const N: usize> std::cmp::PartialEq<$var> for $crate::dim::Fixed<N> {
                #[inline]
                fn eq(&self, v: &$var) -> bool {
                    N == v.0
                }
            }

            impl<B: $crate::Dim> std::ops::Add<B> for $var {
                type Output = $crate::dim::Plus<Self, B>;
                #[inline]
                fn add(self, rhs: B) -> Self::Output {
                    $crate::dim::Plus::new(self, rhs)
                }
            }
            impl<const N: usize> std::ops::Add<$var> for $crate::dim::Fixed<N> {
                type Output = $crate::dim::Plus<$crate::dim::Fixed<N>, $var>;
                #[inline]
                fn add(self, rhs: $var) -> Self::Output {
                    $crate::dim::Plus::new(self, rhs)
                }
            }
        }
        var_dim::$var($var)
    }};
}
