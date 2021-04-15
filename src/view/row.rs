use std::ops::Index;

use super::View;
use crate::{dim, dim::Dim, Matrix};

pub trait RowView<'a>: View<'a, M = dim!(1)> {}

impl<'a, T: 'a, V> RowView<'a> for V where V: View<'a, M = dim!(1), Entry = T> {}

#[derive(Debug)]
pub struct RowSlice<'a, T, M, N> {
    mat: &'a Matrix<T, M, N>,
    i: usize,
}

impl<'a, T, M: Dim, N: Dim> RowSlice<'a, T, M, N> {
    pub(crate) fn new(mat: &'a Matrix<T, M, N>, i: usize) -> Self {
        assert!(i < mat.m.dim());
        Self { mat, i }
    }
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

    #[inline]
    fn get(&self, i: usize, j: usize) -> Option<&'a T> {
        if i == 0 && j < self.n().dim() {
            Some(&self.mat.items[self.i * self.n().dim() + j])
        } else {
            None
        }
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
