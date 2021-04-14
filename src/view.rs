use std::ops::Index;

use crate::{dim, Dim, Matrix};

/// A view into (slice of) a matrix. Or, a reference to a matrix.
///
/// Any view should consist of a refernce to a matrix, and little more.
/// It must be cheap to copy.
/// # Examples
pub trait View<'a>: Sized + Copy
where
    Self: Index<usize, Output = Self::Entry> + Index<[usize; 2], Output = Self::Entry>,
{
    type M: Dim;
    type N: Dim;
    type Entry: 'a;

    fn m(&self) -> Self::M;
    fn n(&self) -> Self::N;

    type Iter: Iterator<Item = &'a Self::Entry>;
    fn iter(&self) -> Self::Iter;

    /// Checks if the two views are equal to one another.
    /// # Examples
    /// ```
    /// # use safemat::prelude::*;
    /// let m = Matrix::from_array([
    ///     [1, 2, 3],
    ///     [2, 3, 4],
    ///     [3, 4, 5]
    /// ]);
    /// let a = m.row_at(1);
    /// let b = m.col_at(1);
    /// assert!(a.equal(b.transpose_ref()));
    /// ```
    fn equal<'b, V>(&self, rhs: V) -> bool
    where
        V: View<'b>,
        Self::Entry: PartialEq<V::Entry>,
        Self::M: PartialEq<V::M>,
        Self::N: PartialEq<V::N>,
    {
        if self.m() != rhs.m() || self.n() != rhs.n() {
            return false;
        }
        self.iter().zip(rhs.iter()).all(|(a, b)| a == b)
    }

    /// Clones each entry in this view into a new owned [`Matrix`].
    /// ```
    /// # use safemat::{*, view::View};
    /// let a = Matrix::from_array([
    ///     [1, 2,  3,  4 ],
    ///     [5, 6,  7,  8 ],
    ///     [9, 10, 11, 12]
    /// ]);
    ///
    /// let b = a.row_at(1).to_matrix();
    /// assert_eq!(b, mat![5, 6, 7, 8]);
    ///
    /// let c = a.col_at(2).to_matrix();
    /// assert_eq!(c, mat![3; 7; 11]);
    /// ```
    fn to_matrix(self) -> Matrix<Self::Entry, Self::M, Self::N>
    where
        Self::Entry: Clone,
    {
        Matrix::try_from_iter_with_dim(self.m(), self.n(), self.iter().cloned()).unwrap()
    }

    /// Gets an interface for accessing a transposed form of this matrix;
    /// does not actually rearrange any data.
    #[inline]
    fn transpose_ref(self) -> Transpose<Self::M, Self::N, Self> {
        Transpose {
            m: self.m(),
            _n: self.n(),
            view: self,
        }
    }
}

/// Interchanges the indices when accessing a matrix view.
/// ```
/// # use safemat::prelude::*;
/// let a = Matrix::from_array([
///     [1, 2, 3, 4],
///     [5, 6, 6, 7],
/// ]);
/// let t = a.as_view().transpose_ref();
/// assert_eq!(t[0], 1);
/// assert_eq!(t[1], 5);
/// assert_eq!(t[2], 2);
///
/// assert_eq!(t[[0,0]], 1);
/// assert_eq!(t[[1,0]], 2);
/// assert_eq!(t[[2,0]], 3);
/// assert_eq!(t[[3,0]], 4);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Transpose<M, N, V> {
    view: V,
    m: M,
    _n: N,
}

impl<'a, V> View<'a> for Transpose<V::M, V::N, V>
where
    V: View<'a>,
{
    type M = V::N;
    type N = V::M;
    type Entry = V::Entry;

    #[inline]
    fn m(&self) -> V::N {
        self.view.n()
    }
    #[inline]
    fn n(&self) -> V::M {
        self.view.m()
    }

    type Iter = V::Iter;
    #[inline]
    fn iter(&self) -> Self::Iter {
        self.view.iter()
    }
}

impl<T, M, N, V: Index<usize>> Index<usize> for Transpose<M, N, V>
where
    M: Dim,
    N: Dim,
    V: Index<[usize; 2], Output = T>,
{
    type Output = T;
    #[inline]
    fn index(&self, idx: usize) -> &T {
        let i = idx / self.m.dim();
        let j = idx % self.m.dim();
        &self.view[[j, i]]
    }
}
impl<T, M, N, V> Index<[usize; 2]> for Transpose<M, N, V>
where
    V: Index<[usize; 2], Output = T>,
{
    type Output = T;
    #[inline]
    fn index(&self, [i, j]: [usize; 2]) -> &T {
        &self.view[[j, i]]
    }
}

impl<'a, T, M: Dim, N: Dim> View<'a> for &'a Matrix<T, M, N> {
    type M = M;
    type N = N;
    type Entry = T;

    #[inline]
    fn m(&self) -> M {
        self.m
    }
    #[inline]
    fn n(&self) -> N {
        self.n
    }

    type Iter = crate::iter::Iter<'a, T, M, N>;
    #[inline]
    fn iter(&self) -> Self::Iter {
        Matrix::iter(self)
    }
}

/// A type from which you can obtain a [`View`] of a specific size.
/// This is implemented by [`Matrix`] as well as every `View` type.
pub trait IntoView<'a>: Sized + 'a {
    type Entry: 'a;
    type M: Dim;
    type N: Dim;
    type View: View<'a, M = Self::M, N = Self::N, Entry = Self::Entry> + 'a;
    /// Gets a [`View`] from this instance.
    ///
    /// If the implementing type is already a `View`,
    /// it should simply return itself.
    fn into_view(self) -> Self::View;
}

impl<'a, V: View<'a> + 'a> IntoView<'a> for V {
    type M = V::M;
    type N = V::N;
    type Entry = V::Entry;
    type View = Self;
    #[inline]
    fn into_view(self) -> Self {
        self
    }
}

pub trait RowView<'a>: View<'a, M = dim!(1)> {}

impl<'a, T: 'a, V> RowView<'a> for V where V: View<'a, M = dim!(1), Entry = T> {}

pub trait IntoRowView<'a>: IntoView<'a, M = dim!(1)> {}

impl<'a, I> IntoRowView<'a> for I where I: IntoView<'a, M = dim!(1)> {}

#[derive(Debug)]
pub struct RowSlice<'a, T, M, N> {
    mat: &'a Matrix<T, M, N>,
    i: usize,
}

impl<T, M, N> Copy for RowSlice<'_, T, M, N> {}

impl<T, M, N> Clone for RowSlice<'_, T, M, N> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            mat: self.mat,
            i: self.i,
        }
    }
}

impl<'a, T, M: Dim, N: Dim> View<'a> for RowSlice<'a, T, M, N> {
    type M = dim!(1);
    type N = N;
    type Entry = T;

    #[inline]
    fn m(&self) -> dim!(1) {
        dim!(1)
    }
    #[inline]
    fn n(&self) -> N {
        self.mat.n
    }

    type Iter = std::slice::Iter<'a, T>;
    fn iter(&self) -> Self::Iter {
        let a = self.i * self.mat.n.dim();
        let b = a + self.mat.n.dim();
        self.mat.items[a..b].iter()
    }
}

impl<T, M: Dim, N: Dim> Index<usize> for RowSlice<'_, T, M, N> {
    type Output = T;
    #[inline]
    fn index(&self, j: usize) -> &T {
        &self.mat[[self.i, j]]
    }
}
impl<T, M: Dim, N: Dim> Index<[usize; 2]> for RowSlice<'_, T, M, N> {
    type Output = T;
    #[inline]
    fn index(&self, [i, j]: [usize; 2]) -> &T {
        assert_eq!(i, 0);
        &self.mat[[self.i, j]]
    }
}

impl<'a, 'b, T, M: Dim, N: Dim, Rhs> PartialEq<Rhs> for RowSlice<'a, T, M, N>
where
    Rhs: View<'b>,
    T: PartialEq<Rhs::Entry>,
    dim!(1): PartialEq<Rhs::M>,
    N: PartialEq<Rhs::N>,
{
    #[inline]
    fn eq(&self, rhs: &Rhs) -> bool {
        self.equal(*rhs)
    }
}

pub trait ColumnView<'a>: View<'a, N = dim!(1)> {}

impl<'a, T: 'a, V> ColumnView<'a> for V where V: View<'a, N = dim!(1), Entry = T> {}

#[derive(Debug)]
pub struct ColumnSlice<'a, T, M, N> {
    mat: &'a Matrix<T, M, N>,
    j: usize,
}

impl<T, M, N> Copy for ColumnSlice<'_, T, M, N> {}

impl<T, M, N> Clone for ColumnSlice<'_, T, M, N> {
    fn clone(&self) -> Self {
        Self {
            mat: self.mat,
            j: self.j,
        }
    }
}

impl<'a, T, M: Dim, N: Dim> View<'a> for ColumnSlice<'a, T, M, N> {
    type M = M;
    type N = dim!(1);
    type Entry = T;

    #[inline]
    fn m(&self) -> M {
        self.mat.m
    }
    #[inline]
    fn n(&self) -> dim!(1) {
        dim!(1)
    }

    type Iter = ColumnSliceIter<'a, T, M, N>;
    fn iter(&self) -> Self::Iter {
        ColumnSliceIter {
            mat: self.mat,
            j: self.j,
            i: 0,
        }
    }
}

impl<T, M: Dim, N: Dim> Index<usize> for ColumnSlice<'_, T, M, N> {
    type Output = T;
    #[inline]
    fn index(&self, i: usize) -> &T {
        &self.mat[[i, self.j]]
    }
}

impl<T, M: Dim, N: Dim> Index<[usize; 2]> for ColumnSlice<'_, T, M, N> {
    type Output = T;
    #[inline]
    fn index(&self, [i, j]: [usize; 2]) -> &T {
        assert_eq!(j, 0);
        &self.mat[[i, self.j]]
    }
}

impl<'a, 'b, T, M: Dim, N: Dim, Rhs> PartialEq<Rhs> for ColumnSlice<'a, T, M, N>
where
    Rhs: View<'b>,
    T: PartialEq<Rhs::Entry>,
    M: PartialEq<Rhs::M>,
    dim!(1): PartialEq<Rhs::N>,
{
    #[inline]
    fn eq(&self, rhs: &Rhs) -> bool {
        self.equal(*rhs)
    }
}

pub struct ColumnSliceIter<'a, T, M, N> {
    mat: &'a Matrix<T, M, N>,
    j: usize,
    i: usize,
}

impl<'a, T, M: Dim, N: Dim> Iterator for ColumnSliceIter<'a, T, M, N> {
    type Item = &'a T;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let i = self.i;
        if i < self.mat.m.dim() {
            self.i += 1;
            Some(&self.mat.items[i * self.mat.n.dim() + self.j])
        } else {
            None
        }
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }
}

impl<T, M: Dim, N: Dim> ExactSizeIterator for ColumnSliceIter<'_, T, M, N> {
    #[inline]
    fn len(&self) -> usize {
        self.mat.m.dim() - self.i
    }
}

impl<T, M: Dim, N: Dim> Matrix<T, M, N> {
    #[inline]
    pub fn as_view(&self) -> &Self {
        self
    }

    pub fn row_at(&self, i: usize) -> RowSlice<'_, T, M, N> {
        assert!(i < self.m.dim());
        RowSlice { mat: self, i }
    }

    pub fn col_at(&self, j: usize) -> ColumnSlice<'_, T, M, N> {
        assert!(j < self.n.dim());
        ColumnSlice { mat: self, j }
    }
}
