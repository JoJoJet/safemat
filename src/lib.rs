use std::ops::{Index, IndexMut};

pub use num_traits::{One, Zero};

pub mod dim;
pub use dim::Dim;
use dim::Fixed;

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
    M1: Dim + PartialEq<M2>,
    M2: Dim,
    N1: Dim + PartialEq<N2>,
    N2: Dim,
{
    fn eq(&self, rhs: &Matrix<T, M2, N2>) -> bool {
        debug_assert_eq!(self.m.dim(), rhs.m.dim());
        debug_assert_eq!(self.n.dim(), rhs.n.dim());
        self.items.eq(&rhs.items)
    }
}
impl<T: Eq, M: Dim + Eq, N: Dim + Eq> Eq for Matrix<T, M, N> {}

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

pub mod ops;

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
