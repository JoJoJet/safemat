use std::ops::Index;

use super::{entry::Entry, View};
use crate::{dim, dim::Dim, Matrix};

pub trait ColumnView<'a>: View<'a, N = dim!(1)> {}

impl<'a, T: 'a, V> ColumnView<'a> for V where V: View<'a, N = dim!(1), Entry = T> {}

#[derive(Debug)]
pub struct ColumnSlice<'a, T, M, N> {
    mat: &'a Matrix<T, M, N>,
    j: usize,
}

impl<'a, T, M: Dim, N: Dim> ColumnSlice<'a, T, M, N> {
    pub(crate) fn new(mat: &'a Matrix<T, M, N>, j: usize) -> Self {
        assert!(j < mat.n.dim());
        Self { mat, j }
    }

    /// Gets an iterator over the [`Entry`]s in this column slice.
    #[inline]
    pub fn entries(self) -> RowsIter<'a, T, M, N> {
        RowsIter {
            mat: self.mat,
            j: self.j,
            i: 0,
        }
    }
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

    #[inline]
    fn get(&self, i: usize, j: usize) -> Option<&'a T> {
        if j == 0 && i < self.m().dim() {
            Some(&self.mat.items[i * self.mat.n.dim() + self.j])
        } else {
            None
        }
    }

    type Iter = ColumnSliceIter<'a, T, M, N>;
    #[inline]
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

pub struct RowsIter<'a, T, M, N> {
    mat: &'a Matrix<T, M, N>,
    j: usize,
    i: usize,
}

impl<'a, T, M: Dim, N: Dim> Iterator for RowsIter<'a, T, M, N> {
    type Item = Entry<'a, T, M, N>;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let i = self.i;
        if i < self.mat.m.dim() {
            self.i += 1;
            Some(Entry::new(self.mat, i, self.j))
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
impl<T, M: Dim, N: Dim> ExactSizeIterator for RowsIter<'_, T, M, N> {
    #[inline]
    fn len(&self) -> usize {
        self.mat.m.dim() - self.i
    }
}
