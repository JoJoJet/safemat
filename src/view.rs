use std::ops::Index;

use crate::{dim, dim::Identity, Dim, Matrix};

/// A view into (2D slice of) a matrix. Or, a reference to a matrix.
/// # Examples
/// Views support linear indexing.
/// ```
/// # use safemat::*;
/// let m = Matrix::from_array([
///     [1,  2,  3,  4 ],
///     [5,  6,  7,  8 ],
///     [9,  10, 11, 12],
///     [13, 14, 15, 16]
/// ]);
/// let v = m.view(1, 1, dim!(2), dim!(2));
/// assert_eq!(v[0], 6);
/// assert_eq!(v[1], 7);
/// assert_eq!(v[2], 10);
/// assert_eq!(v[3], 11);
/// ```
///
/// As well as square indexing.
/// ```
/// # use safemat::*;
/// let m = Matrix::from_array([
///     [1,  2,  3,  4,  5 ],
///     [6,  7,  8,  9,  10],
///     [11, 12, 13, 14, 15]
/// ]);
/// let v = m.view(1, 1, dim!(2), dim!(3));
/// assert_eq!(v[[0, 0]], 7);
/// assert_eq!(v[[0, 1]], 8);
/// assert_eq!(v[[0, 2]], 9);
/// assert_eq!(v[[1, 0]], 12);
/// assert_eq!(v[[1, 1]], 13);
/// assert_eq!(v[[1, 2]], 14);
/// ```
pub trait View<'a, T: 'a, M: Dim, N: Dim>
where
    Self: Index<usize, Output = T> + Index<[usize; 2], Output = T>,
{
    fn m(&self) -> M;
    fn n(&self) -> N;

    type Iter: Iterator<Item = &'a T>;
    fn iter(&self) -> Self::Iter;
}

/// A view into a matrix; a reference to a matrix.
///
/// # Type parameters:
/// `Ms`, `Ns`, the dimensions of this view.
///
/// `Mo`, `No`, the dimensions of the original matrix.
///
/// For examples, see [`View`].
#[derive(Debug)]
pub struct MatrixView<'mat, T, Ms, Ns, Mo, No> {
    mat: &'mat Matrix<T, Mo, No>,
    m: Ms,
    n: Ns,
    i: usize,
    j: usize,
}

use crate::iter::MatrixViewIter;
impl<'a, T: 'a, Ms: Dim, Ns: Dim, Mo: Dim, No: Dim> View<'a, T, Ms, Ns>
    for MatrixView<'a, T, Ms, Ns, Mo, No>
{
    #[inline]
    fn m(&self) -> Ms {
        self.m
    }
    #[inline]
    fn n(&self) -> Ns {
        self.n
    }
    type Iter = MatrixViewIter<'a, T, Ms, Ns, Mo, No>;
    fn iter(&self) -> Self::Iter {
        MatrixViewIter {
            mat: self.mat,
            m: self.m,
            n: self.n,
            i: self.i,
            j: self.j,
            i_iter: 0,
            j_iter: 0,
        }
    }
}

impl<T, Ms: Dim, Ns: Dim, Mo: Dim, No: Dim> Index<usize> for MatrixView<'_, T, Ms, Ns, Mo, No> {
    type Output = T;
    #[inline]
    fn index(&self, idx: usize) -> &Self::Output {
        assert!(idx < self.m.dim() * self.n.dim());
        let i = self.i + idx / self.n.dim();
        let j = self.j + idx % self.n.dim();
        debug_assert!(i < self.mat.m.dim());
        debug_assert!(j < self.mat.n.dim());
        &self.mat.items[i * self.mat.n.dim() + j]
    }
}

impl<T, Ms: Dim, Ns: Dim, Mo: Dim, No: Dim> Index<[usize; 2]>
    for MatrixView<'_, T, Ms, Ns, Mo, No>
{
    type Output = T;
    #[inline]
    fn index(&self, [i, j]: [usize; 2]) -> &Self::Output {
        assert!(i < self.m.dim());
        assert!(j < self.n.dim());
        let i = self.i + i;
        let j = self.j + j;
        debug_assert!(i < self.mat.m.dim());
        debug_assert!(j < self.mat.n.dim());
        &self.mat.items[i * self.mat.n.dim() + j]
    }
}

impl<'a, T, Ms, Ns, Mo, No, Rhs> PartialEq<Rhs> for MatrixView<'a, T, Ms, Ns, Mo, No>
where
    T: PartialEq,
    Ms: Dim,
    Ns: Dim,
    Mo: Dim,
    No: Dim,
    Rhs: IntoView<'a, T, Ms, Ns> + Clone,
{
    fn eq(&self, rhs: &Rhs) -> bool {
        self.iter()
            .zip(rhs.clone().into_view().iter())
            .all(|(a, b)| a == b)
    }
}

impl<'a, T, M: Dim, N: Dim, M2: Identity<M>, N2: Identity<N>> View<'a, T, M, N> for &'a Matrix<T, M2, N2> {
    #[inline]
    fn m(&self) -> M {
        self.m.identity()
    }
    #[inline]
    fn n(&self) -> N {
        self.n.identity()
    }

    type Iter = crate::iter::Iter<'a, T, M2, N2>;
    #[inline]
    fn iter(&self) -> Self::Iter {
        Matrix::iter(self)
    }
}

/// A type from which you can obtain a [`View`] of a specific size.
/// This is implemented by [`Matrix`] as well as every `View` type.
pub trait IntoView<'a, T: 'a, M: Dim, N: Dim>: Sized + 'a {
    type View: View<'a, T, M, N> + 'a;
    /// Gets a [`View`] from this instance.
    ///
    /// If the implementing type is already a `View`,
    /// it should simply return itself.
    fn into_view(self) -> Self::View;
}

impl<'a, T, M, N, Mo, No> IntoView<'a, T, M, N> for MatrixView<'a, T, M, N, Mo, No>
where
    M: Dim,
    N: Dim,
    Mo: Dim,
    No: Dim,
{
    type View = Self;
    fn into_view(self) -> Self {
        self
    }
}

impl<'a, T, M: Dim, N: Dim, M2: Identity<M>, N2: Identity<N>> IntoView<'a, T, M, N> for &'a Matrix<T, M2, N2> {
    type View = Self;
    #[inline]
    fn into_view(self) -> Self::View {
        self
    }
}

impl<T, M: Dim, N: Dim> Matrix<T, M, N> {
    /// Gets a view into this matrix.
    /// # Parameters
    /// `i`: the vertical index at which to start the view.
    ///
    /// `j`: the horizontal index at which to start the view.
    ///
    /// `len_m`: the vertical length of the view.
    ///
    /// `len_n`: the horizontal length of the view.
    #[inline]
    pub fn view<Ms: Dim, Ns: Dim>(
        &self,
        i: usize,
        j: usize,
        len_m: Ms,
        len_n: Ns,
    ) -> MatrixView<'_, T, Ms, Ns, M, N> {
        assert!(i + len_m.dim() <= self.m.dim());
        assert!(j + len_n.dim() <= self.n.dim());
        MatrixView {
            mat: self,
            m: len_m,
            n: len_n,
            i,
            j,
        }
    }
}

/// A view into a row of a matrix.
/// ```
/// # use safemat::*;
/// let mat1 = mat![ 1, 2, 3 ; 4, 5, 6 ];
/// // RowViews can be compared with slices.
/// assert_eq!(mat1.row_at(1), [4, 5, 6].as_ref());
///
/// // They can also be compared with other RowViews.
/// let mat2 = mat![ 0, 1, 0 ; 1, 2, 3 ; 2, 3, 4 ];
/// assert_eq!(mat1.row_at(0), mat2.row_at(1));
/// assert_ne!(mat1.row_at(1), mat2.row_at(1));
///
/// // Or even with ColumnViews.
/// assert_eq!(mat1.row_at(0), mat2.column_at(1));
/// assert_ne!(mat1.row_at(1), mat2.column_at(0));
///
/// // RowViews are also iterators.
/// let mut row = mat1.row_at(0);
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
    M1: Dim,
    M2: Dim,
    N1: Dim + PartialEq<N2>,
    N2: Dim,
{
    fn eq(&self, rhs: &RowView<'_, T, M2, N2>) -> bool {
        let n = self.mat.n.dim();
        if n != rhs.mat.n.dim() {
            return false;
        }
        let a = self.i * n;
        let b = rhs.i * n;
        &self.mat.items[a..a + n] == &rhs.mat.items[b..b + n]
    }
}

impl<T, M1, M2, N1, N2> PartialEq<ColumnView<'_, T, M2, N2>> for RowView<'_, T, M1, N1>
where
    T: PartialEq,
    M1: Dim,
    M2: Dim,
    N1: Dim + PartialEq<M2>,
    N2: Dim,
{
    fn eq(&self, rhs: &ColumnView<'_, T, M2, N2>) -> bool {
        let n = self.mat.n.dim();
        if n != rhs.mat.m.dim() {
            return false;
        }
        let i = self.i * n;
        rhs.eq(&&self.mat.items[i..i + n])
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
/// let mat2 = mat![ 0, 1, 0 ; 1, 3, 5 ; 0, 5, 0 ];
/// assert_eq!(mat.column_at(0), mat2.column_at(1));
/// assert_ne!(mat.column_at(1), mat2.column_at(0));
///
/// // Or even with `RowView`s.
/// assert_eq!(mat.column_at(0), mat2.row_at(1));
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
    M1: Dim + PartialEq<M2>,
    M2: Dim,
    N1: Dim,
    N2: Dim,
{
    fn eq(&self, rhs: &ColumnView<'_, T, M2, N2>) -> bool {
        let m = self.mat.m.dim();
        if m != rhs.mat.m.dim() {
            return false;
        }
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

impl<T, M1, M2, N1, N2> PartialEq<RowView<'_, T, M2, N2>> for ColumnView<'_, T, M1, N1>
where
    T: PartialEq,
    M1: Dim + PartialEq<N2>,
    M2: Dim,
    N1: Dim,
    N2: Dim,
{
    fn eq(&self, rhs: &RowView<'_, T, M2, N2>) -> bool {
        let n = rhs.mat.n.dim();
        let b = rhs.i * n;
        self.eq(&&rhs.mat.items[b..b + n])
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
