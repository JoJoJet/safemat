use std::{
    iter::Sum,
    ops::{Index, IndexMut, Mul},
};

pub use num_traits::{One, Zero};

mod dim;
pub use dim::{Dim, Fixed, FixedDim, Plus};

pub mod view;

pub mod iter;

#[derive(Clone, Debug)]
pub struct Matrix<T, M, N> {
    m: M,
    n: N,
    items: Box<[T]>,
}

impl<T, M1, M2, N1, N2> PartialEq<Matrix<T, M2, N2>> for Matrix<T, M1, N1>
where
    T: PartialEq,
    M1: Dim,
    M2: Dim + Into<M1>,
    N1: Dim,
    N2: Dim + Into<N1>,
{
    fn eq(&self, rhs: &Matrix<T, M2, N2>) -> bool {
        debug_assert_eq!(self.m.dim(), rhs.m.dim());
        debug_assert_eq!(self.n.dim(), rhs.n.dim());
        self.items.eq(&rhs.items)
    }
}
impl<T: Eq, M: Dim, N: Dim> Eq for Matrix<T, M, N> {}

impl<T, M: Dim, N: Dim> Matrix<T, M, N> {
    #[inline]
    pub fn default_with_dim(m: M, n: N) -> Self
    where
        T: Default,
    {
        Self::from_fn_with_dim(m, n, |_, _| T::default())
    }
    pub fn from_fn_with_dim(m: M, n: N, mut f: impl FnMut(usize, usize) -> T) -> Self {
        let mu = m.dim();
        let nu = n.dim();
        let len = mu * nu;
        let mut items = Vec::with_capacity(len);
        items.extend(
            (0..mu)
                .flat_map(|m| (0..nu).map(move |n| (m, n)))
                .map(|(m, n)| f(m, n)),
        );
        debug_assert_eq!(items.len(), len);
        Self {
            m,
            n,
            items: items.into_boxed_slice(),
        }
    }

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

    /// Performs an infallible conversion on the M dimension of this matrix.
    /// ```
    /// # use safemat::*;
    /// let a = mat![1];
    /// let b = mat![3];
    /// let c = a.vcat(b);
    /// // `c` is currently a column vector with length Plus<1, 1>,
    /// // but we want the length to just be 2. `patch_m` can help.
    /// let c: FixedVec<_, 2> = c.patch_m();
    /// ```
    /// Note that this will fail if the dimensions don't have an infallible conversion:
    /// ```compile_fail
    /// let a = mat![1];
    /// let b = mat![3];
    /// let c = a.vcat(b);
    /// let c: FixedVec<_, 3> = c.patch_m(); // this fails
    /// ```
    #[inline]
    pub fn patch_m<M2: Dim>(self) -> Matrix<T, M2, N>
    where
        M: Into<M2>,
    {
        Matrix {
            m: self.m.into(),
            n: self.n,
            items: self.items,
        }
    }

    #[inline]
    pub fn patch_n<N2: Dim>(self) -> Matrix<T, M, N2>
    where
        N: Into<N2>,
    {
        Matrix {
            m: self.m,
            n: self.n.into(),
            items: self.items,
        }
    }

    /// Coerces M to a known fixed dimension.
    pub fn coerce_m<M2: FixedDim>(self) -> Result<Matrix<T, M2, N>, CoerceError> {
        if self.m.dim() == M2::DIM {
            Ok(Matrix {
                m: M2::new(),
                n: self.n,
                items: self.items,
            })
        } else {
            Err(CoerceError::M(self.m.dim(), M2::DIM))
        }
    }

    pub fn coerce_n<N2: FixedDim>(self) -> Result<Matrix<T, M, N2>, CoerceError> {
        if self.n.dim() == N2::DIM {
            Ok(Matrix {
                m: self.m,
                n: N2::new(),
                items: self.items,
            })
        } else {
            Err(CoerceError::N(self.n.dim(), N2::DIM))
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
            let rhs = Matrix::<U, N, N2> {
                m: self.n,
                n: rhs.n,
                items: rhs.items,
            };
            Ok(&self * &rhs)
        } else {
            Err(MulError(self.n.dim(), rhs.m.dim()))
        }
    }

    /// Vertically concatenates the two matrices.
    /// ```
    /// # use safemat::*;
    /// let a = mat![ 1, 2 ; 3, 4 ];
    /// let b = mat![ 5, 6 ];
    /// let c = a.vcat(b);
    /// assert_eq!(c, mat![ 1, 2 ; 3, 4 ; 5, 6 ]);
    /// ```
    pub fn vcat<M2: Dim>(self, other: Matrix<T, M2, N>) -> Matrix<T, Plus<M, M2>, N> {
        let mut items = Vec::from(self.items);
        items.extend(Vec::from(other.items).into_iter());
        let m = Plus(self.m, other.m);
        let n = self.n;
        assert_eq!(items.len(), m.dim() * n.dim());
        Matrix {
            m: Plus(self.m, other.m),
            n: self.n,
            items: items.into_boxed_slice(),
        }
    }

    /// Horizontally concatenates the two matrices.
    /// ```
    /// # use safemat::*;
    /// let a = mat![ 1 ; 2 ; 3 ];
    /// let b = mat![ 4 ; 5 ; 6 ];
    /// let c = a.hcat(b);
    /// assert_eq!(c, mat![ 1, 4 ; 2, 5 ; 3, 6 ]);
    /// ```
    pub fn hcat<N2: Dim>(self, other: Matrix<T, M, N2>) -> Matrix<T, M, Plus<N, N2>> {
        let len = self.m.dim() * (self.n.dim() + other.n.dim());
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
            n: Plus(self.n, other.n),
            items: items.into_boxed_slice(),
        }
    }
}

impl<T: Zero + One, N: Dim> Matrix<T, N, N> {
    /// Returns the identity matrix of specified size.
    ///
    /// ```
    /// # use safemat::*;
    /// let i = Matrix::id_with_dim(dim!(3));
    /// assert_eq!(i, Matrix::from_array([
    ///     [1, 0, 0],
    ///     [0, 1, 0],
    ///     [0, 0, 1],
    /// ]));
    /// ```
    pub fn id_with_dim(n: N) -> Self {
        Self::from_fn_with_dim(n, n, |i, j| if i == j { T::one() } else { T::zero() })
    }
}

#[derive(Debug, thiserror::Error)]
#[error("lhs n: {0}, rhs m: {1}")]
pub struct MulError(usize, usize);

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum CoerceError {
    #[error("dimension m, {0}, not compatible with target {1}")]
    M(usize, usize),
    #[error("dimension n, {0}, not compatible with target {1}")]
    N(usize, usize),
}

impl<T, M: Dim, N: Dim> Index<[usize; 2]> for Matrix<T, M, N> {
    type Output = T;
    fn index(&self, [i, j]: [usize; 2]) -> &T {
        assert!(i < self.m.dim());
        assert!(j < self.n.dim());
        &self.items[i * self.n.dim() + j]
    }
}
impl<T, M: Dim, N: Dim> IndexMut<[usize; 2]> for Matrix<T, M, N> {
    fn index_mut(&mut self, [i, j]: [usize; 2]) -> &mut T {
        assert!(i < self.m.dim());
        assert!(j < self.n.dim());
        &mut self.items[i * self.n.dim() + j]
    }
}

impl<'a, T, U, V, M: Dim, K: Dim, K2: Dim + Into<K>, N: Dim> Mul<&'a Matrix<U, K2, N>>
    for &'a Matrix<T, M, K>
where
    V: Sum,
    for<'b> &'b T: Mul<&'b U, Output = V>,
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

pub type FixedMat<T, const M: usize, const N: usize> = Matrix<T, Fixed<M>, Fixed<N>>;

impl<T: Default, const M: usize, const N: usize> Default for FixedMat<T, M, N> {
    #[inline]
    fn default() -> Self {
        Self::default_with_dim(Fixed::<M>, Fixed::<N>)
    }
}

impl<T, const M: usize, const N: usize> FixedMat<T, M, N> {
    #[inline]
    pub fn from_fn(f: impl FnMut(usize, usize) -> T) -> Self {
        Self::from_fn_with_dim(Fixed::<M>, Fixed::<N>, f)
    }

    pub fn from_array(ary: [[T; N]; M]) -> Self {
        use std::array::IntoIter;
        let mut items = Vec::with_capacity(M * N);
        items.extend(IntoIter::new(ary).flat_map(IntoIter::new));
        debug_assert_eq!(items.len(), M * N);
        Self {
            m: Fixed::<M>,
            n: Fixed::<N>,
            items: items.into_boxed_slice(),
        }
    }
}

/// Type alias for a column vector.
pub type Vector<T, M> = Matrix<T, M, dim!(1)>;
/// Type alias for a column vector with fixed length.
pub type FixedVec<T, const M: usize> = Vector<T, Fixed<M>>;

/// Provides something similar to a macro literal syntax.
/// Rows are delineated with a semicolon ;
///
/// ```
/// # use safemat::*;
/// let a = mat![1, 2, 3 ; 4, 5, 6];
/// let b = Matrix::from_array([
///     [1, 2, 3],
///     [4, 5, 6],
/// ]);
/// assert_eq!(a, b);
/// ```
#[macro_export]
macro_rules! mat {
    ($($($itm: expr),*);*) => {
        Matrix::from_array([
            $([$($itm),*]),*
            ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mat_macro() {
        let a = mat![1, 2, 3 ; 4, 5, 6];
        assert_eq!(a, Matrix::from_array([[1, 2, 3], [4, 5, 6]]));
    }

    #[test]
    fn from_ary() {
        let mat = Matrix::from_array([[1, 2, 3], [4, 5, 6]]);

        assert_eq!(mat[[0, 0]], 1);
        assert_eq!(mat[[0, 1]], 2);
        assert_eq!(mat[[0, 2]], 3);
        assert_eq!(mat[[1, 0]], 4);
        assert_eq!(mat[[1, 1]], 5);
        assert_eq!(mat[[1, 2]], 6);
    }

    #[test]
    fn transpose() {
        let mat = mat![1, 2, 3 ; 4, 5, 6];
        eprintln!("{:?}", mat);

        let mat = mat.transpose();
        eprintln!("{:?}", mat);

        assert_eq!(mat[[0, 0]], 1);
        assert_eq!(mat[[1, 0]], 2);
        assert_eq!(mat[[2, 0]], 3);
        assert_eq!(mat[[0, 1]], 4);
        assert_eq!(mat[[1, 1]], 5);
        assert_eq!(mat[[2, 1]], 6);
    }

    #[test]
    fn mul_fixed() {
        let a = mat![1, 2, 3];
        let b = mat![4; 5; 6];

        let c = &a * &b;
        assert_eq!(c, mat![32]);
    }

    #[test]
    fn vcat() {
        let a = mat![1, 2, 3];
        let b = mat![4, 5, 6];

        let c = a.vcat(b);
        assert_eq!(c, mat![1, 2, 3 ; 4, 5, 6]);
    }
}
