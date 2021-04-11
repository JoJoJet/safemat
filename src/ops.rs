use std::{
    iter::Sum,
    ops::{Add, Mul},
};

use crate::{
    dim::{Fixed, Identity},
    Dim, Matrix,
};

#[derive(Debug, thiserror::Error)]
#[error("lhs n: {0}, rhs m: {1}")]
pub struct MulError(usize, usize);

impl<T, M: Dim, N: Dim> Matrix<T, M, N> {
    /// Returns the transpose of this matrix.
    ///
    /// ```
    /// # use safemat::*;
    /// let a = Matrix::from_array([
    ///     [1, 2, 3],
    ///     [4, 5, 6],
    /// ]);
    /// let b = a.transpose();
    /// assert_eq!(b, Matrix::from_array([
    ///     [1, 4],
    ///     [2, 5],
    ///     [3, 6],    
    /// ]));
    /// ```
    pub fn transpose(self) -> Matrix<T, N, M> {
        let mut result = Matrix::<Option<T>, _, _>::default_with_dim(self.n, self.m);
        for (i, j, itm) in self.into_iter().indices() {
            result[[j, i]] = Some(itm);
        }
        result.map(|_, _, itm| itm.unwrap())
    }

    /// Applies a transforming function to every element of this matrix.
    ///
    /// ```
    /// # use safemat::*;
    /// let a = Matrix::from_array([
    ///     [1, 2, 3],
    ///     [4, 5, 6],
    /// ]);
    /// let b = a.map(|_, _, itm| itm as f32 + 2.0);
    /// assert_eq!(b, Matrix::from_array([
    ///     [3.0, 4.0, 5.0],
    ///     [6.0, 7.0, 8.0],
    /// ]));
    /// ```
    pub fn map<U, F: FnMut(usize, usize, T) -> U>(self, mut f: F) -> Matrix<U, M, N> {
        let m = self.m;
        let n = self.n;
        let mut items = Vec::with_capacity(m.dim() * n.dim());
        items.extend(self.into_iter().indices().map(|(i, j, itm)| f(i, j, itm)));
        debug_assert_eq!(items.len(), m.dim() * n.dim());
        Matrix {
            m,
            n,
            items: items.into_boxed_slice(),
        }
    }

    /// Tries to convert `M` to a known fixed dimension.
    /// ```
    /// # use safemat::*;
    /// let len = 4; // a variable with some unknown value.
    /// let mat = Matrix::from_fn_with_dim(dim!(len), dim!(1), |i, _| i+1);
    /// // We don't know how tall it is, so let's try to convert its `M` dimension.
    /// match mat.try_m::<2>() {
    ///     Ok(mat) => panic!("this shouldn't have worked, pardner..."),
    ///     // It didn't work, so lets try again.
    ///     Err(mat) => match mat.try_m::<4>() {
    ///         Ok(_) => println!("howdy"),
    ///         Err(_) => panic!("i have no earthly idea what this is"),
    ///     }
    /// }
    /// ```
    pub fn fix_m<const M2: usize>(self) -> Result<Matrix<T, Fixed<M2>, N>, Self> {
        if self.m.dim() == M2 {
            Ok(Matrix {
                m: Fixed,
                n: self.n,
                items: self.items,
            })
        } else {
            Err(self)
        }
    }

    /// Tries to convert `N` to a known fixed dimension.
    pub fn fix_n<const N2: usize>(self) -> Result<Matrix<T, M, Fixed<N2>>, Self> {
        if self.n.dim() == N2 {
            Ok(Matrix {
                m: self.m,
                n: Fixed,
                items: self.items,
            })
        } else {
            Err(self)
        }
    }

    /// Vertically concatenates the two matrices.
    /// ```
    /// # use safemat::*;
    /// let a = mat![ 1, 2 ; 3, 4 ];
    /// let b = mat![ 5, 6 ];
    /// let c: FixedMat<_, 3, 2> = a.vcat(b); // Type annotations not necessary.
    /// assert_eq!(c, mat![ 1, 2 ; 3, 4 ; 5, 6 ]);
    /// ```
    /// It also works for variable-length matrices.
    /// ```
    /// # use safemat::{*, dim::*};
    /// # fn main() {
    /// # do_vcat(Some("3".to_owned()).into_iter()).unwrap();
    /// # }
    /// # fn do_vcat(mut args: impl Iterator<Item = String>) -> Result<(), Box<dyn std::error::Error>> {
    /// let len = args.next() // Grab a number from the command line.
    ///     .ok_or("no args")?
    ///     .parse::<usize>()?;
    ///
    /// let a = Matrix::from_fn_with_dim(dim!(len), dim!(2), |i, j| i+j);
    /// let b = mat![ 5, 9 ];
    /// let c: Matrix<_, Plus<_,dim!(1)>, _> = a.vcat(b); // length = `len` + 1
    /// # assert_eq!(c, mat![ 0, 1 ; 1, 2 ; 2, 3 ; 5, 9 ]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn vcat<M2, N2>(self, other: Matrix<T, M2, N2>) -> Matrix<T, <M as Add<M2>>::Output, N>
    where
        M: Add<M2>,
        <M as Add<M2>>::Output: Dim,
        M2: Dim,
        N2: Identity<N>,
    {
        assert_eq!(self.n.dim(), other.n.dim());
        let mut items = Vec::from(self.items);
        items.extend(Vec::from(other.items).into_iter());
        let m = self.m + other.m;
        let n = self.n;
        assert_eq!(items.len(), m.dim() * n.dim());
        Matrix {
            m,
            n: self.n,
            items: items.into_boxed_slice(),
        }
    }

    /// Horizontally concatenates the two matrices.
    /// ```
    /// # use safemat::*;
    /// let a = mat![ 1 ; 2 ; 3 ];
    /// let b = mat![ 4 ; 5 ; 6 ];
    /// let c: FixedMat<_, 3, 2> = a.hcat(b); // Type annotations not necessary.
    /// assert_eq!(c, mat![ 1, 4 ; 2, 5 ; 3, 6 ]);
    /// ```
    pub fn hcat<M2, N2>(self, other: Matrix<T, M2, N2>) -> Matrix<T, M, <N as Add<N2>>::Output>
    where
        M2: Identity<M>,
        N: Add<N2>,
        <N as Add<N2>>::Output: Dim,
        N2: Dim,
    {
        assert_eq!(self.m.dim(), other.m.dim());
        let n = self.n + other.n;
        let len = self.m.dim() * n.dim();
        let mut items = Vec::with_capacity(len);
        let mut a = Vec::from(self.items).into_iter();
        let mut b = Vec::from(other.items).into_iter();
        for _ in 0..self.m.dim() {
            for _ in 0..self.n.dim() {
                items.push(a.next().unwrap());
            }
            for _ in 0..other.n.dim() {
                items.push(b.next().unwrap());
            }
        }
        assert_eq!(items.len(), len);
        Matrix {
            m: self.m,
            n,
            items: items.into_boxed_slice(),
        }
    }

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
