use super::*;

/// A view into a row of a matrix.
/// ```
/// # use safemat::*;
/// let mat = mat![ 1, 2, 3 ; 4, 5, 6 ];
/// let row = mat.row_at(1);
/// assert_eq!(row, [4, 5, 6].as_ref());
/// ```
#[derive(Debug, Clone)]
pub struct RowView<'a, T, M, N> {
    pub(crate) mat: &'a Matrix<T, M, N>,
    pub(crate) i: usize,
}

impl<T, M1, M2, N1, N2> PartialEq<RowView<'_, T, M2, N2>> for RowView<'_, T, M1, N1>
where
    T: PartialEq,
    N1: Dim,
    N2: Dim + Into<N1>,
{
    fn eq(&self, rhs: &RowView<'_, T, M2, N2>) -> bool {
        let n = self.mat.n.dim();
        assert_eq!(n, rhs.mat.n.dim());
        let a = self.i * n;
        let b = rhs.i * n;
        &self.mat.items[a..a + n] == &rhs.mat.items[b..b + n]
    }
}

impl<T: PartialEq, M, N: Dim> PartialEq<&'_ [T]> for RowView<'_, T, M, N> {
    fn eq(&self, rhs: &&[T]) -> bool {
        let n = self.mat.n.dim();
        let i = self.i * n;
        &self.mat.items[i..i + n] == *rhs
    }
}

impl<T, M: Dim, N: Dim> Index<usize> for RowView<'_, T, M, N> {
    type Output = T;
    fn index(&self, j: usize) -> &T {
        let n = self.mat.n.dim();
        assert!(j < n);
        &self.mat.items[self.i * n + j]
    }
}

impl<T, M: Dim, N: Dim> Matrix<T, M, N> {
    /// Gets a reference to the row at specified index.
    #[inline]
    pub fn row_at(&self, i: usize) -> RowView<'_, T, M, N> {
        assert!(i < self.m.dim());
        RowView { mat: self, i }
    }
}
