use std::{
    iter::Sum,
    ops::{Add, Mul},
};

use crate::{
    dim::{Fixed, Identity},
    view::{IntoRowView, IntoView, View},
    Dim, Matrix, RowVec,
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
    #[inline]
    pub fn map<U, F: FnMut(usize, usize, T) -> U>(self, mut f: F) -> Matrix<U, M, N> {
        Matrix::try_from_iter_with_dim(
            self.m,
            self.n,
            self.into_iter().indices().map(|(i, j, itm)| f(i, j, itm)),
        )
        .unwrap()
    }

    /// Tries to convert `M` to a known fixed dimension.
    /// ```
    /// # use safemat::*;
    /// let len = 4; // a variable with some unknown value.
    /// let mat = Matrix::from_fn_with_dim(dim!(len), dim!(1), |i, _| i+1);
    /// // We don't know how tall it is, so let's try to convert its `M` dimension.
    /// match mat.fix_m::<2>() {
    ///     Ok(mat) => panic!("this shouldn't have worked, pardner..."),
    ///     // It didn't work, so lets try again.
    ///     Err(mat) => match mat.fix_m::<4>() {
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
impl<T, U, V, M, K1, K2, N> Mul<&'_ Matrix<U, K2, N>> for &'_ Matrix<T, M, K1>
where
    V: Sum,
    for<'a, 'b> &'a T: Mul<&'b U, Output = V>,
    M: Dim,
    K1: Dim,
    K2: Identity<K1>,
    N: Dim,
{
    type Output = Matrix<V, M, N>;
    #[inline]
    fn mul(self, rhs: &Matrix<U, K2, N>) -> Self::Output {
        let k = self.n.dim();
        assert_eq!(k, rhs.m.dim());
        Matrix::from_fn_with_dim(self.m, rhs.n, |i, j| {
            (0..k).map(|k| &self[[i, k]] * &rhs[[k, j]]).sum()
        })
    }
}

/// ```
/// # use safemat::*;
/// let a = mat![ 1, 2, 3 ; 4, 5, 6 ];
/// let b = mat![ 4, 5, 6 ; 7, 8, 9 ];
/// let c = a + b;
/// assert_eq!(c, mat![ 5, 7, 9 ; 11, 13, 15 ]);
/// ```
/// Matrix dimensions must agree in order to do addition.
/// ```compile_fail
/// use safemat::*;
/// let a = mat![ 1, 2, 3 ; 4, 5, 6 ];
/// let b = mat![ 4, 5, 6 ];
/// let c = a + b; // this fails.
/// ```
impl<T, U, V, M1, M2, N1, N2> Add<Matrix<U, M2, N2>> for Matrix<T, M1, N1>
where
    T: Add<U, Output = V>,
    M1: Dim,
    M2: Identity<M1>,
    N1: Dim,
    N2: Identity<N1>,
{
    type Output = Matrix<V, M1, N1>;
    #[inline]
    fn add(self, rhs: Matrix<U, M2, N2>) -> Self::Output {
        assert_eq!(self.m.dim(), rhs.m.dim());
        assert_eq!(self.n.dim(), rhs.n.dim());
        Matrix::try_from_iter_with_dim(
            self.m,
            self.n,
            self.into_iter().zip(rhs.into_iter()).map(|(t, u)| t + u),
        )
        .unwrap()
    }
}

/// Addition by reference. This only works for heterogenous types.
/// ```
/// # use safemat::*;
/// let a = mat![ 1, 2, 3 ];
/// let b = mat![ 2, 2, 2 ];
/// let c = &a + &b;
/// assert_eq!(c, mat![ 3, 4, 5 ]);
/// ```
impl<'a, T, V, M, N, Rhs> Add<Rhs> for &'a Matrix<T, M, N>
where
    Rhs: IntoView<'a, T, M, N>,
    for<'b> &'a T: Add<&'b T, Output = V>,
    M: Dim,
    N: Dim,
{
    type Output = Matrix<V, M, N>;
    #[inline]
    fn add(self, rhs: Rhs) -> Self::Output {
        let rhs = rhs.into_view();
        assert_eq!(self.m.dim(), rhs.m().dim());
        assert_eq!(self.n.dim(), rhs.n().dim());
        Matrix::try_from_iter_with_dim(
            self.m,
            self.n,
            self.iter().zip(rhs.iter()).map(|(t, u)| t + u),
        )
        .unwrap()
    }
}

impl<T, M: Dim, N: Dim> Matrix<T, M, N> {
    /// Adds a scalar value to each entry in this matrix.
    /// ```
    /// # use safemat::*;
    /// let a = mat![ 1, 2 ; 3, 4 ];
    /// let b = a.add_scalar(2);
    /// assert_eq!(b, mat![ 3, 4 ; 5, 6 ]);
    /// ```
    #[inline]
    pub fn add_scalar<U, V>(self, u: U) -> Matrix<V, M, N>
    where
        T: Add<U, Output = V>,
        U: Clone,
    {
        Matrix::try_from_iter_with_dim(self.m, self.n, self.into_iter().map(|t| t + u.clone()))
            .unwrap()
    }

    /// Adds a scalar value to each entry in this matrix,
    /// then returns a new matrix without moving anything.
    /// ```
    /// # use safemat::*;
    /// let a = mat![ 1, 2 ; 3, 4 ];
    /// let b = a.add_scalar_ref(&1);
    /// assert_eq!(b, mat![ 2, 3 ; 4, 5 ]);
    /// ```
    #[inline]
    pub fn add_scalar_ref<U, V>(&self, u: &U) -> Matrix<V, M, N>
    where
        for<'a, 'b> &'a T: Add<&'b U, Output = V>,
    {
        Matrix::try_from_iter_with_dim(self.m, self.n, self.iter().map(|t| t + u)).unwrap()
    }

    /// Adds the specified row vector to each row of this matrix.
    /// ```
    /// # use safemat::*;
    /// let a = Matrix::from_array([
    ///     [ 1, 2, 3 ],
    ///     [ 4, 5, 6 ],
    ///     [ 7, 8, 9 ]
    /// ]);
    /// let r = mat![ 1, 2, 3 ];
    ///
    /// assert_eq!(a.add_row(r), Matrix::from_array([
    ///     [ 2, 4,  6  ],
    ///     [ 5, 7,  9  ],
    ///     [ 8, 10, 12 ]
    /// ]));
    /// ```
    #[inline]
    pub fn add_row<U, V, N2>(self, row: RowVec<U, N2>) -> Matrix<V, M, N>
    where
        T: Add<U, Output = V>,
        U: Clone,
        N2: Identity<N>,
    {
        assert_eq!(self.n.dim(), row.n.dim());
        Matrix::try_from_iter_with_dim(
            self.m,
            self.n,
            self.into_rows()
                .map(|r1| r1 + row.clone())
                .flat_map(IntoIterator::into_iter),
        )
        .unwrap()
    }

    /// Adds the specified row vector to each row of this matrix,
    /// returning a new matrix without moving anything
    /// ```
    /// # use safemat::*;
    /// let a = Matrix::from_array([
    ///     [ 1, 2, 3 ],
    ///     [ 4, 5, 6 ],
    /// ]);
    /// let r = mat![ 1, 2, 3 ];
    ///
    /// assert_eq!(a.add_row_ref(&r), Matrix::from_array([
    ///     [ 2, 4, 6 ],
    ///     [ 5, 7, 9 ],
    /// ]));
    /// ```
    pub fn add_row_ref<'a, U: 'a, V>(&'a self, row: impl IntoRowView<'a, U, N>) -> Matrix<V, M, N>
    where
        for<'b> &'a T: Add<&'b U, Output = V>,
    {
        let row = row.into_view();
        let n = self.n.dim();
        assert_eq!(n, row.n().dim());
        Matrix::try_from_iter_with_dim(
            self.m,
            self.n,
            self.rows()
                .flat_map(|r1| r1.iter().zip(row.iter()).map(|(t, u)| t + u)),
        )
        .unwrap()
    }
}
