use std::{iter::Sum, ops::Mul};

use crate::{dim::Identity, Dim, Matrix};

#[derive(Debug, thiserror::Error)]
#[error("lhs n: {0}, rhs m: {1}")]
pub struct MulError(usize, usize);

impl<T, M: Dim, N: Dim> Matrix<T, M, N> {
    pub fn try_mul<U, V, N2: Dim>(
        self,
        rhs: Matrix<U, impl Dim, N2>,
    ) -> Result<Matrix<V, M, N2>, MulError>
    where
        V: Sum,
        for<'a> &'a T: Mul<&'a U, Output = V>,
    {
        if self.n.dim() == rhs.m.dim() {
            let k = self.n.dim();
            Ok(Matrix::from_fn_with_dim(self.m, rhs.n, |i, j| {
                (0..k).map(|k| &self[[i, k]] * &rhs[[k, j]]).sum()
            }))
        } else {
            Err(MulError(self.n.dim(), rhs.m.dim()))
        }
    }
}

/// The `N` dimension of the lhs must agree with the `M` dimension of the rhs.
/// ```compile_fail
/// # use safemat::*;
/// let k = 3;
/// let a = Matrix::<i32, _, _>::id_with_dim(dim!(k));
/// let b = mat![ 1, 2 ; 3, 4 ; 5, 6 ];
/// let c = &a * &b; // This fails because dimension `k`
///                  // is not known to agree with `3` at compile-time.
/// ```
///
/// ```
/// # use safemat::*;
/// let k = 3;
/// let k = dim!(k);
/// let a = Matrix::<i32, _, _>::id_with_dim(k);
/// let b = Matrix::from_fn_with_dim(k, dim!(2), |i, j| (i + j) as i32);
/// let c = &a * &b;
/// ```
impl<'a, T, U, V, M, K1, K2, N> Mul<&'a Matrix<U, K2, N>> for &'a Matrix<T, M, K1>
where
    V: Sum,
    for<'b> &'b T: Mul<&'b U, Output = V>,
    M: Dim,
    K1: Dim,
    K2: Identity<K1>,
    N: Dim,
{
    type Output = Matrix<V, M, N>;
    #[inline]
    fn mul(self, rhs: &'a Matrix<U, K2, N>) -> Self::Output {
        assert_eq!(self.n.dim(), rhs.m.dim());
        let k = self.n.dim();
        Matrix::from_fn_with_dim(self.m, rhs.n, |i, j| {
            (0..k).map(|k| &self[[i, k]] * &rhs[[k, j]]).sum()
        })
    }
}
