use std::{
    iter::Sum,
    ops::{Index, IndexMut, Mul},
};

pub use num_traits::{One, Zero};

mod dim;
pub use dim::{Dim, Fixed, FixedDim, Patch, Plus};

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

    /// Patches the composite dimension/s into respective singular dimensions.
    /// If either dimension does not implement [`Patch`], see [`patch_m`] or [`patch_n`].
    /// ```
    /// # use safemat::*;
    /// let m = mat![1];
    /// let m = m.vcat(mat![2]);
    /// let m = m.hcat(mat![ 3 , 4 ; 5, 6 ]);
    /// // The matrix currently has dimensions `Plus<1, 1>` x `Plus<1, 2>`,
    /// // but we'd rather it just be 2 x 3.
    /// // `patch` can help.
    /// let m: FixedMat<_, 2, 3> = m.patch(); // Type annotations not necessary.
    /// ```
    #[inline]
    pub fn patch(self) -> Matrix<T, M::Target, N::Target>
    where
        M: Patch,
        N: Patch,
    {
        let m = self.m.patch();
        let n = self.n.patch();
        debug_assert_eq!(self.m.dim(), m.dim());
        debug_assert_eq!(self.n.dim(), n.dim());
        Matrix {
            m,
            n,
            items: self.items,
        }
    }

    /// Patches the composite dimension `M` into a single dimension.
    /// ```
    /// # use safemat::*;
    /// let a = mat![1];
    /// let b = mat![3];
    /// let c = a.vcat(b);
    /// // `c` is currently a column vector with length Plus<1, 1>,
    /// // but we want the length to just be 2.
    /// // `patch_m` can help.
    /// let c: FixedVec<_, 2> = c.patch_m(); // Type annotations not necessary.
    /// ```
    #[inline]
    pub fn patch_m(self) -> Matrix<T, M::Target, N>
    where
        M: Patch,
    {
        let m = self.m.patch();
        debug_assert_eq!(self.m.dim(), m.dim());
        Matrix {
            m,
            n: self.n,
            items: self.items,
        }
    }

    /// Patches the composite dimension `n` into a single dimesnion.
    #[inline]
    pub fn patch_n(self) -> Matrix<T, M, N::Target>
    where
        N: Patch,
    {
        let n = self.n.patch();
        debug_assert_eq!(self.n.dim(), n.dim());
        Matrix {
            m: self.m,
            n,
            items: self.items,
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
    pub fn try_m<const M2: usize>(self) -> Result<Matrix<T, Fixed<M2>, N>, Self> {
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
    pub fn try_n<const N2: usize>(self) -> Result<Matrix<T, M, Fixed<N2>>, Self> {
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
    pub fn vcat<M2, N2>(self, other: Matrix<T, M2, N2>) -> Matrix<T, Plus<M, M2>, N>
    where
        M2: Dim,
        N2: Dim + Into<N>,
    {
        assert_eq!(self.n.dim(), other.n.dim());
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
    pub fn hcat<M2, N2>(self, other: Matrix<T, M2, N2>) -> Matrix<T, M, Plus<N, N2>>
    where
        M2: Dim + Into<M>,
        N2: Dim,
    {
        assert_eq!(self.m.dim(), other.m.dim());
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

impl<T, M: Dim, N: Dim> Index<usize> for Matrix<T, M, N> {
    type Output = T;
    #[inline]
    fn index(&self, i: usize) -> &T {
        &self.items[i]
    }
}
impl<T, M: Dim, N: Dim> IndexMut<usize> for Matrix<T, M, N> {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut T {
        &mut self.items[i]
    }
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

pub type RowVec<T, N> = Matrix<T, dim!(1), N>;
pub type FixedRowVec<T, const N: usize> = RowVec<T, Fixed<N>>;

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
