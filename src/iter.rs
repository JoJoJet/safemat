use super::*;

use std::{iter::FusedIterator, marker::PhantomData};

/// A matrix iterator that internally tracks its indices.
/// You should be able to access these with an inherent `indices` method.
#[doc(hidden)]
pub trait IndexIterator: Iterator {
    /// Retrives the indices corresponding to the next item
    /// yielded from this iterator.
    /// If the next item yielded will be `None`, this can return a garbage value.
    fn get_indices(&self) -> (usize, usize);

    fn get_indices_back(&self) -> (usize, usize)
    where
        Self: DoubleEndedIterator;
}

/// An iterator over a matrix that yields the indices of the current element.
/// ```
/// # use safemat::*;
/// let mat = mat![ 1, 2 ; 3, 4 ];
/// let mut i = mat.iter().indices();
/// assert_eq!(i.next(), Some((0, 0, &1)));
/// assert_eq!(i.next(), Some((0, 1, &2)));
/// assert_eq!(i.next(), Some((1, 0, &3)));
/// assert_eq!(i.next(), Some((1, 1, &4)));
/// assert_eq!(i.next(), None);
/// ```
pub struct Indices<I: IndexIterator>(pub(crate) I);

impl<I: IndexIterator> Iterator for Indices<I> {
    type Item = (usize, usize, I::Item);
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let (i, j) = self.0.get_indices();
        Some((i, j, self.0.next()?))
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}
impl<I> DoubleEndedIterator for Indices<I>
where
    I: IndexIterator + DoubleEndedIterator,
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        let (i, j) = self.0.get_indices_back();
        Some((i, j, self.0.next_back()?))
    }
}

impl<I> ExactSizeIterator for Indices<I>
where
    I: IndexIterator + ExactSizeIterator,
{
    #[inline]
    fn len(&self) -> usize {
        self.0.len()
    }
}

impl<I: IndexIterator + FusedIterator> FusedIterator for Indices<I> {}

/// Projects a linear index into a pair of 2D indices.
#[inline]
fn project2(n: impl Dim, idx: usize) -> (usize, usize) {
    (idx / n.dim(), idx % n.dim())
}

/// An iterator that moves out of a matrix.
/// ```
/// # use safemat::*;
/// let mat = mat![ 1 ; 2 ];
/// let mut i = mat.into_iter();
/// assert_eq!(i.next(), Some(1));
/// assert_eq!(i.next(), Some(2));
/// assert_eq!(i.next(), None);
/// ```
pub struct IntoIter<T, M: Dim, N: Dim> {
    _m: PhantomData<M>,
    n: N,
    iter: std::vec::IntoIter<T>,
    count: usize,
}

impl<T, M: Dim, N: Dim> IntoIter<T, M, N> {
    #[inline]
    pub fn indices(self) -> Indices<Self> {
        Indices(self)
    }
}

impl<T, M: Dim, N: Dim> IndexIterator for IntoIter<T, M, N> {
    #[inline]
    fn get_indices(&self) -> (usize, usize) {
        project2(self.n, self.count)
    }
    #[inline]
    fn get_indices_back(&self) -> (usize, usize) {
        project2(self.n, self.iter.len() + self.count)
    }
}

impl<T, M: Dim, N: Dim> Iterator for IntoIter<T, M, N> {
    type Item = T;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.count += 1;
        self.iter.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<T, M: Dim, N: Dim> DoubleEndedIterator for IntoIter<T, M, N> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.next_back()
    }
}

impl<T, M: Dim, N: Dim> ExactSizeIterator for IntoIter<T, M, N> {}

/// An iterator over the references to the items in a matrix.
/// To get the indices of each item, see the `indices` method.
/// ```
/// # use safemat::*;
/// let m = mat![ 1, 2 ; 3, 4 ];
/// let mut i = m.iter();
/// assert_eq!(i.next(), Some(&1));
/// assert_eq!(i.next(), Some(&2));
/// assert_eq!(i.next(), Some(&3));
/// assert_eq!(i.next(), Some(&4));
/// assert_eq!(i.next(), None);
/// ```
pub struct Iter<'a, T, M: Dim, N: Dim> {
    _m: PhantomData<M>,
    n: N,
    iter: std::slice::Iter<'a, T>,
    count: usize,
}

impl<'a, T, M: Dim, N: Dim> Iter<'a, T, M, N> {
    #[inline]
    pub fn indices(self) -> Indices<Self> {
        Indices(self)
    }
}

impl<T, M: Dim, N: Dim> IndexIterator for Iter<'_, T, M, N> {
    #[inline]
    fn get_indices(&self) -> (usize, usize) {
        project2(self.n, self.count)
    }
    #[inline]
    fn get_indices_back(&self) -> (usize, usize) {
        project2(self.n, self.iter.len() + self.count)
    }
}

impl<'a, T, M: Dim, N: Dim> Iterator for Iter<'a, T, M, N> {
    type Item = &'a T;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.count += 1;
        self.iter.next()
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<T, M: Dim, N: Dim> DoubleEndedIterator for Iter<'_, T, M, N> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.next_back()
    }
}

impl<T, M: Dim, N: Dim> ExactSizeIterator for Iter<'_, T, M, N> {}

/// An iterator over mutable references of a matrix.
/// ```
/// # use safemat::*;
/// let mut m = Matrix::from_array([
///     [ 1, 2, 3 ],
///     [ 4, 5, 6 ],
///     [ 7, 8, 9 ],
/// ]);
/// for (i, j, itm) in m.iter_mut().indices() {
///    *itm += i * j;
/// }
/// assert_eq!(m, Matrix::from_array([
///     [ 1, 2,  3  ],
///     [ 4, 6,  8  ],
///     [ 7, 10, 13 ],
/// ]));
/// ```
pub struct IterMut<'a, T, M: Dim, N: Dim> {
    _m: PhantomData<M>,
    n: N,
    iter: std::slice::IterMut<'a, T>,
    count: usize,
}

impl<'a, T, M: Dim, N: Dim> IterMut<'a, T, M, N> {
    #[inline]
    pub fn indices(self) -> Indices<Self> {
        Indices(self)
    }
}

impl<T, M: Dim, N: Dim> IndexIterator for IterMut<'_, T, M, N> {
    #[inline]
    fn get_indices(&self) -> (usize, usize) {
        project2(self.n, self.count)
    }
    #[inline]
    fn get_indices_back(&self) -> (usize, usize) {
        project2(self.n, self.iter.len() + self.count)
    }
}

impl<'a, T, M: Dim, N: Dim> Iterator for IterMut<'a, T, M, N> {
    type Item = &'a mut T;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.count += 1;
        self.iter.next()
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<T, M: Dim, N: Dim> DoubleEndedIterator for IterMut<'_, T, M, N> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.next_back()
    }
}

impl<T, M: Dim, N: Dim> ExactSizeIterator for IterMut<'_, T, M, N> {}

/// An iterator that splits a matrix into a set of owned rows.
/// ```
/// # use safemat::*;
/// let mat = mat![ 1, 2, 3, 4 ; 5, 6, 7, 8 ];
/// let mut rows = mat.into_rows();
/// assert_eq!(rows.next(), Some(mat![1, 2, 3, 4]));
/// assert_eq!(rows.next(), Some(mat![5, 6, 7, 8]));
/// assert_eq!(rows.next(), None);
/// ```
pub struct IntoRows<T, M, N> {
    m: M,
    n: N,
    iter: std::vec::IntoIter<T>,
    i: usize,
}

impl<T, M: Dim, N: Dim> Iterator for IntoRows<T, M, N> {
    type Item = RowVec<T, N>;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.i < self.m.dim() {
            self.i += 1;
            Some(RowVec::from_fn_with_dim(dim!(1), self.n, |_, _| {
                self.iter.next().unwrap()
            }))
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

impl<T, M: Dim, N: Dim> ExactSizeIterator for IntoRows<T, M, N> {
    #[inline]
    fn len(&self) -> usize {
        self.m.dim() - self.i
    }
}

impl<T, M: Dim, N: Dim> FusedIterator for IntoRows<T, M, N> {}

/// An iterator over the rows of a matrix.
/// ```
/// # use safemat::prelude::*;
/// let mat = mat![ 1, 2 ; 3, 4 ; 5, 6 ];
/// let mut i = mat.rows();
/// assert_eq!(i.next().unwrap(), &mat![1, 2]);
/// assert_eq!(i.next().unwrap(), &mat![3, 4]);
/// assert_eq!(i.next().unwrap(), &mat![5, 6]);
/// assert!(i.next().is_none());
/// ```
pub struct Rows<'a, T, M: Dim, N: Dim> {
    mat: &'a Matrix<T, M, N>,
    i: usize,
}

impl<'a, T, M: Dim, N: Dim> Iterator for Rows<'a, T, M, N> {
    type Item = view::row::RowSlice<'a, T, M, N>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.i < self.mat.m.dim() {
            let i = self.i;
            self.i += 1;
            Some(self.mat.row_at(i))
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

impl<T, M: Dim, N: Dim> ExactSizeIterator for Rows<'_, T, M, N> {
    #[inline]
    fn len(&self) -> usize {
        self.mat.m.dim().checked_sub(self.i).unwrap_or(0)
    }
}

impl<T, M: Dim, N: Dim> FusedIterator for Rows<'_, T, M, N> {}

/// An iterator that splits a matrix into a set of owned columns.
/// ```
/// # use safemat::*;
/// let mat = mat![ 1, 2, 3, 4 ; 5, 6, 7, 8 ];
/// let mut cols = mat.into_columns();
/// assert_eq!(cols.next(), Some(mat![1 ; 5]));
/// assert_eq!(cols.next(), Some(mat![2 ; 6]));
/// assert_eq!(cols.next(), Some(mat![3 ; 7]));
/// assert_eq!(cols.next(), Some(mat![4 ; 8]));
/// assert_eq!(cols.next(), None);
/// ```
pub struct IntoColumns<T, M, N> {
    mat: Matrix<Option<T>, M, N>,
    j: usize,
}

impl<T, M: Dim, N: Dim> Iterator for IntoColumns<T, M, N> {
    type Item = Vector<T, M>;
    fn next(&mut self) -> Option<Self::Item> {
        let j = self.j;
        let n = self.mat.n.dim();
        if j < n {
            self.j += 1;
            Some(Vector::from_fn_with_dim(self.mat.m, dim!(1), |i, _| {
                self.mat.items[i * n + j].take().unwrap()
            }))
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

impl<T, M: Dim, N: Dim> ExactSizeIterator for IntoColumns<T, M, N> {
    #[inline]
    fn len(&self) -> usize {
        self.mat.n.dim() - self.j
    }
}

impl<T, M: Dim, N: Dim> FusedIterator for IntoColumns<T, M, N> {}

/// An iterator over the columns of a matrix.
/// ```
/// # use safemat::*;
/// let mat = mat![ 1, 2, 3 ; 4, 5, 6 ; 7, 8, 9 ];
/// let mut i = mat.columns();
/// assert_eq!(i.next().unwrap(), &mat![1; 4; 7]);
/// assert_eq!(i.next().unwrap(), &mat![2; 5; 8]);
/// assert_eq!(i.next().unwrap(), &mat![3; 6; 9]);
/// assert!(i.next().is_none());
/// ```
pub struct Columns<'a, T, M: Dim, N: Dim> {
    mat: &'a Matrix<T, M, N>,
    j: usize,
}

impl<'a, T, M: Dim, N: Dim> Iterator for Columns<'a, T, M, N> {
    type Item = crate::view::col::ColumnSlice<'a, T, M, N>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.j < self.mat.n.dim() {
            let j = self.j;
            self.j += 1;
            Some(self.mat.col_at(j))
        } else {
            None
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }
}

impl<T, M: Dim, N: Dim> ExactSizeIterator for Columns<'_, T, M, N> {
    fn len(&self) -> usize {
        self.mat.n.dim().checked_sub(self.j).unwrap_or(0)
    }
}

impl<T, M: Dim, N: Dim> FusedIterator for Columns<'_, T, M, N> {}

impl<T, M: Dim, N: Dim> IntoIterator for Matrix<T, M, N> {
    type Item = T;
    type IntoIter = IntoIter<T, M, N>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        IntoIter {
            _m: PhantomData,
            n: self.n,
            iter: Vec::from(self.items).into_iter(),
            count: 0,
        }
    }
}

impl<'a, T, M: Dim, N: Dim> IntoIterator for &'a Matrix<T, M, N> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T, M, N>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        Iter {
            _m: PhantomData,
            n: self.n,
            iter: self.items.iter(),
            count: 0,
        }
    }
}

impl<'a, T, M: Dim, N: Dim> IntoIterator for &'a mut Matrix<T, M, N> {
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T, M, N>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        IterMut {
            _m: PhantomData,
            n: self.n,
            iter: self.items.iter_mut(),
            count: 0,
        }
    }
}

impl<T, M: Dim, N: Dim> Matrix<T, M, N> {
    #[inline]
    pub fn iter(&self) -> Iter<'_, T, M, N> {
        self.into_iter()
    }

    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<'_, T, M, N> {
        self.into_iter()
    }

    #[inline]
    pub fn rows(&self) -> Rows<'_, T, M, N> {
        Rows { mat: self, i: 0 }
    }

    #[inline]
    pub fn into_rows(self) -> IntoRows<T, M, N> {
        IntoRows {
            m: self.m,
            n: self.n,
            iter: Vec::from(self.items).into_iter(),
            i: 0,
        }
    }

    #[inline]
    pub fn columns(&self) -> Columns<'_, T, M, N> {
        Columns { mat: self, j: 0 }
    }

    #[inline]
    pub fn into_columns(self) -> IntoColumns<T, M, N> {
        IntoColumns {
            mat: self.map(|_, _, itm| Some(itm)),
            j: 0,
        }
    }
}
