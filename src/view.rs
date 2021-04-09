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

/// A view into a column of a matrix.
/// ```
/// # use safemat::*;
/// let mat = mat![ 1, 2 ; 3, 4 ; 5, 6 ];
/// // `ColumnView`s can be compared with slices.
/// assert_eq!(mat.column_at(1), [2, 4, 6].as_ref());
/// assert_ne!(mat.column_at(0), [1, 3, 5, 7].as_ref());
///
/// // They can also be compared with other `ColumnView`s.
/// let mat2 = mat![ 0, 1 ; 0, 3 ; 0, 5 ];
/// assert_eq!(mat.column_at(0), mat2.column_at(1));
/// assert_ne!(mat.column_at(1), mat2.column_at(0));
/// ```
#[derive(Debug, Clone)]
pub struct ColumnView<'a, T, M, N> {
    pub(crate) mat: &'a Matrix<T, M, N>,
    pub(crate) j: usize,
}

impl<T, M1, M2, N1, N2> PartialEq<ColumnView<'_, T, M2, N2>> for ColumnView<'_, T, M1, N1>
where
    T: PartialEq,
    M1: Dim,
    M2: Dim + Into<M1>,
    N1: Dim,
    N2: Dim,
{
    fn eq(&self, rhs: &ColumnView<'_, T, M2, N2>) -> bool {
        let m = self.mat.m.dim();
        assert_eq!(m, rhs.mat.m.dim());
        let n1 = self.mat.n.dim();
        let n2 = rhs.mat.n.dim();
        for i in 0..m {
            if self.mat.items[i * n1 + self.j] != rhs.mat.items[i * n2 + rhs.j] {
                return false;
            }
        }
        return true
    }
}

impl<T: PartialEq, M: Dim, N: Dim> PartialEq<&'_ [T]> for ColumnView<'_, T, M, N> {
    fn eq(&self, rhs: &&[T]) -> bool {
        let m = self.mat.m.dim();
        if m != rhs.len() {
            return false;
        }
        let n = self.mat.n.dim();
        for i in 0..m {
            if self.mat.items[i * n + self.j] != rhs[i] {
                return false;
            }
        }
        return true;
    }
}

impl<T, M: Dim, N: Dim> Matrix<T, M, N> {
    /// Gets a reference to the row at specified index.
    #[inline]
    pub fn row_at(&self, i: usize) -> RowView<'_, T, M, N> {
        assert!(i < self.m.dim());
        RowView { mat: self, i }
    }

    /// Gets a reference to the column at specified index.
    #[inline]
    pub fn column_at(&self, j: usize) -> ColumnView<'_, T, M, N> {
        assert!(j < self.n.dim());
        ColumnView { mat: self, j }
    }
}
