use super::*;
use crate::view::*;

use std::{
    iter::{Enumerate, FusedIterator},
    marker::PhantomData,
};

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
pub struct Indices<I: Iterator, M: Dim, N: Dim> {
    _m: PhantomData<M>,
    n: N,
    iter: Enumerate<I>,
}

impl<I: Iterator, M: Dim, N: Dim> Iterator for Indices<I, M, N> {
    type Item = (usize, usize, I::Item);
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let (idx, itm) = self.iter.next()?;
        let n = self.n.dim();
        Some((idx / n, idx % n, itm))
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}
impl<I, M: Dim, N: Dim> DoubleEndedIterator for Indices<I, M, N>
where
    I: DoubleEndedIterator + ExactSizeIterator,
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        let (idx, itm) = self.iter.next_back()?;
        let n = self.n.dim();
        Some((idx / n, idx % n, itm))
    }
}

impl<I: ExactSizeIterator, M: Dim, N: Dim> ExactSizeIterator for Indices<I, M, N> {
    #[inline]
    fn len(&self) -> usize {
        self.iter.len()
    }
}

impl<I: FusedIterator, M: Dim, N: Dim> FusedIterator for Indices<I, M, N> {}

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
    iter: Enumerate<std::vec::IntoIter<T>>,
}

impl<T, M: Dim, N: Dim> IntoIter<T, M, N> {
    #[inline]
    pub fn indices(self) -> Indices<impl Iterator<Item = T>, M, N> {
        Indices {
            _m: PhantomData,
            n: self.n,
            iter: self.iter,
        }
    }
}

impl<T, M: Dim, N: Dim> Iterator for IntoIter<T, M, N> {
    type Item = T;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|(_, itm)| itm)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<T, M: Dim, N: Dim> DoubleEndedIterator for IntoIter<T, M, N> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.next_back().map(|(_, itm)| itm)
    }
}

impl<T, M: Dim, N: Dim> ExactSizeIterator for IntoIter<T, M, N> {}

/// An iterator over the elements of a [`MatrixView`].
/// ```
/// # use safemat::*;
/// # use safemat::view::View;
/// let m = mat![ 1, 2, 3, 4 ; 5, 6, 7, 8 ; 9, 10, 11, 12 ];
/// let mut i = m.view(1, 1, dim!(2), dim!(2)).iter();
/// assert_eq!(i.next(), Some(&6));
/// assert_eq!(i.next(), Some(&7));
/// assert_eq!(i.next(), Some(&10));
/// assert_eq!(i.next(), Some(&11));
/// assert_eq!(i.next(), None);
/// ```
pub struct MatrixViewIter<'a, T, Ms, Ns, Mo, No> {
    pub(crate) mat: &'a Matrix<T, Mo, No>,
    pub(crate) m: Ms,
    pub(crate) n: Ns,
    pub(crate) i: usize,
    pub(crate) j: usize,
    pub(crate) i_iter: usize,
    pub(crate) j_iter: usize,
}

impl<'a, T, Ms, Ns, Mo, No> Iterator for MatrixViewIter<'a, T, Ms, Ns, Mo, No>
where
    Ms: Dim,
    Ns: Dim,
    Mo: Dim,
    No: Dim,
{
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        let i = self.i_iter;
        if i < self.m.dim() {
            let j = self.j_iter;
            if j < self.n.dim() - 1 {
                self.j_iter += 1;
            } else {
                self.j_iter = 0;
                self.i_iter += 1;
            }
            let i = self.i + i;
            let j = self.j + j;
            Some(&self.mat.items[i * self.mat.n.dim() + j])
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

impl<T, Ms, Ns, Mo, No> ExactSizeIterator for MatrixViewIter<'_, T, Ms, Ns, Mo, No>
where
    Ms: Dim,
    Ns: Dim,
    Mo: Dim,
    No: Dim,
{
    #[inline]
    fn len(&self) -> usize {
        let idx = self.i_iter * self.n.dim() + self.j_iter;
        self.m.dim() * self.n.dim() - idx
    }
}

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
    iter: Enumerate<std::slice::Iter<'a, T>>,
}

impl<'a, T, M: Dim, N: Dim> Iter<'a, T, M, N> {
    #[inline]
    pub fn indices(self) -> Indices<impl Iterator<Item = &'a T>, M, N> {
        Indices {
            _m: PhantomData,
            n: self.n,
            iter: self.iter,
        }
    }
}

impl<'a, T, M: Dim, N: Dim> Iterator for Iter<'a, T, M, N> {
    type Item = &'a T;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|(_, itm)| itm)
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<T, M: Dim, N: Dim> DoubleEndedIterator for Iter<'_, T, M, N> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.next_back().map(|(_, itm)| itm)
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
    iter: Enumerate<std::slice::IterMut<'a, T>>,
}

impl<'a, T, M: Dim, N: Dim> IterMut<'a, T, M, N> {
    #[inline]
    pub fn indices(self) -> Indices<impl Iterator<Item = &'a mut T>, M, N> {
        Indices {
            _m: PhantomData,
            n: self.n,
            iter: self.iter,
        }
    }
}

impl<'a, T, M: Dim, N: Dim> Iterator for IterMut<'a, T, M, N> {
    type Item = &'a mut T;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|(_, itm)| itm)
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<T, M: Dim, N: Dim> DoubleEndedIterator for IterMut<'_, T, M, N> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.next_back().map(|(_, itm)| itm)
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
    type Item = RowViewImpl<'a, T, N, M, N>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.i < self.mat.m.dim() {
            let i = self.i;
            self.i += 1;
            Some(self.mat.view(i, 0, dim!(1), self.mat.n))
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
    type Item = ColumnViewImpl<'a, T, M, M, N>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.j < self.mat.n.dim() {
            let j = self.j;
            self.j += 1;
            Some(self.mat.view(0, j, self.mat.m, dim!(1)))
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
            iter: Vec::from(self.items).into_iter().enumerate(),
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
            iter: self.items.iter().enumerate(),
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
            iter: self.items.iter_mut().enumerate(),
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
