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

/// An iterator over the rows of a matrix.
/// ```
/// # use safemat::*;
/// let mat = mat![ 1, 2 ; 3, 4 ; 5, 6 ];
/// let mut i = mat.rows();
/// assert_eq!(i.next().unwrap(), [1, 2].as_ref());
/// assert_eq!(i.next().unwrap(), [3, 4].as_ref());
/// assert_eq!(i.next().unwrap(), [5, 6].as_ref());
/// assert_eq!(i.next(), None);
/// ```
pub struct Rows<'a, T, M: Dim, N: Dim> {
    mat: &'a Matrix<T, M, N>,
    i: usize,
}

impl<'a, T, M: Dim, N: Dim> Iterator for Rows<'a, T, M, N> {
    type Item = RowView<'a, T, M, N>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.i < self.mat.m.dim() {
            let i = self.i;
            self.i += 1;
            Some(RowView { mat: self.mat, i })
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

/// An iterator over the columns of a matrix.
/// ```
/// # use safemat::*;
/// let mat = mat![ 1, 2, 3 ; 4, 5, 6 ; 7, 8, 9 ];
/// let mut i = mat.columns();
/// assert_eq!(&*i.next().unwrap(), [&1, &4, &7].as_ref());
/// assert_eq!(&*i.next().unwrap(), [&2, &5, &8].as_ref());
/// assert_eq!(&*i.next().unwrap(), [&3, &6, &9].as_ref());
/// assert_eq!(i.next(), None);
/// ```
pub struct Columns<'a, T, M: Dim, N: Dim> {
    mat: &'a Matrix<T, M, N>,
    j: usize,
}

impl<'a, T, M: Dim, N: Dim> Iterator for Columns<'a, T, M, N> {
    type Item = Box<[&'a T]>;
    fn next(&mut self) -> Option<Self::Item> {
        let n = self.mat.n.dim();
        let j = self.j;
        self.j += 1;
        if self.j <= n {
            let mut v = Vec::with_capacity(n);
            for i in 0..n {
                v.push(&self.mat[[i, j]]);
            }
            Some(v.into_boxed_slice())
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
    pub fn columns(&self) -> Columns<'_, T, M, N> {
        Columns { mat: self, j: 0 }
    }
}
