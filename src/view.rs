use super::*;

/// A view into a row of a matrix.
/// ```
/// # use safemat::*;
/// let mat = mat![ 1, 2, 3 ; 4, 5, 6 ];
/// let row = mat.row_at(1);
/// assert_eq!(row, [4, 5, 6].as_ref());
///
/// // RowViews are also iterators.
/// let mut row = mat.row_at(0);
/// assert_eq!(row.next(), Some(&1));
/// assert_eq!(row.next(), Some(&2));
/// assert_eq!(row.next(), Some(&3));
/// assert_eq!(row.next(), None);
/// ```
#[derive(Debug, Clone)]
pub struct RowView<'a, T, M, N> {
    pub(crate) mat: &'a Matrix<T, M, N>,
    pub(crate) i: usize,
    pub(crate) j_iter: usize,
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

impl<'a, T, M, N: Dim> Iterator for RowView<'a, T, M, N> {
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        let j = self.j_iter;
        let n = self.mat.n.dim();
        if j < n {
            self.j_iter += 1;
            Some(&self.mat.items[self.i * n + j])
        } else {
            None
        }
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = ExactSizeIterator::len(self);
        (len, Some(len))
    }
}

impl<T, M, N: Dim> ExactSizeIterator for RowView<'_, T, M, N> {
    #[inline]
    fn len(&self) -> usize {
        self.mat.n.dim() - self.j_iter
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
///
/// // ColumnViews are also iterators
/// let mut col = mat.column_at(0);
/// assert_eq!(col.next(), Some(&1));
/// assert_eq!(col.next(), Some(&3));
/// assert_eq!(col.next(), Some(&5));
/// assert_eq!(col.next(), None);
/// ```
#[derive(Debug, Clone)]
pub struct ColumnView<'a, T, M, N> {
    pub(crate) mat: &'a Matrix<T, M, N>,
    pub(crate) j: usize,
    pub(crate) i_iter: usize,
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
        return true;
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

impl<'a, T, M: Dim, N: Dim> Iterator for ColumnView<'a, T, M, N> {
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        let i = self.i_iter;
        if i < self.mat.m.dim() {
            self.i_iter += 1;
            Some(&self.mat.items[i * self.mat.n.dim() + self.j])
        } else {
            None
        }
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = ExactSizeIterator::len(self);
        (len, Some(len))
    }
}

impl<T, M: Dim, N: Dim> ExactSizeIterator for ColumnView<'_, T, M, N> {
    #[inline]
    fn len(&self) -> usize {
        self.mat.m.dim() - self.i_iter
    }
}

impl<T, M: Dim, N: Dim> Matrix<T, M, N> {
    /// Gets a reference to the row at specified index.
    #[inline]
    pub fn row_at(&self, i: usize) -> RowView<'_, T, M, N> {
        assert!(i < self.m.dim());
        RowView {
            mat: self,
            i,
            j_iter: 0,
        }
    }

    /// Gets a reference to the column at specified index.
    #[inline]
    pub fn column_at(&self, j: usize) -> ColumnView<'_, T, M, N> {
        assert!(j < self.n.dim());
        ColumnView {
            mat: self,
            j,
            i_iter: 0,
        }
    }
}
