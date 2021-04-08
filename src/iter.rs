use super::*;

use std::{iter::Enumerate, marker::PhantomData};

pub struct IntoIter<T, M: Dim, N: Dim> {
    _m: PhantomData<M>,
    n: N,
    iter: Enumerate<std::vec::IntoIter<T>>,
}

impl<T, M: Dim, N: Dim> Iterator for IntoIter<T, M, N> {
    type Item = (usize, usize, T);
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

impl<T, M: Dim, N: Dim> DoubleEndedIterator for IntoIter<T, M, N> {
    #[inline]
    fn next_back(&mut self) -> Option<(usize, usize, T)> {
        let (idx, itm) = self.iter.next_back()?;
        let n = self.n.dim();
        Some((idx / n, idx % n, itm))
    }
}

impl<T, M: Dim, N: Dim> ExactSizeIterator for IntoIter<T, M, N> {}

/// An iterator over the references to the items in a matrix,
/// as well as the respective indices of the items.
/// ```
/// # use safemat::*;
/// let m = mat![ 1, 2 ; 3, 4 ];
/// let mut i = m.iter();
/// assert_eq!(i.next(), Some((0, 0, &1)));
/// assert_eq!(i.next(), Some((0, 1, &2)));
/// assert_eq!(i.next(), Some((1, 0, &3)));
/// assert_eq!(i.next(), Some((1, 1, &4)));
/// assert_eq!(i.next(), None);
/// ```
pub struct Iter<'a, T, M: Dim, N: Dim> {
    _m: PhantomData<M>,
    n: N,
    iter: Enumerate<std::slice::Iter<'a, T>>,
}

impl<'a, T, M: Dim, N: Dim> Iterator for Iter<'a, T, M, N> {
    type Item = (usize, usize, &'a T);
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

impl<T, M: Dim, N: Dim> DoubleEndedIterator for Iter<'_, T, M, N> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        let (idx, itm) = self.iter.next_back()?;
        let n = self.n.dim();
        Some((idx / n, idx % n, itm))
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
/// for (i, j, itm) in &mut m {
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

impl<'a, T, M: Dim, N: Dim> Iterator for IterMut<'a, T, M, N> {
    type Item = (usize, usize, &'a mut T);
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

impl<T, M: Dim, N: Dim> DoubleEndedIterator for IterMut<'_, T, M, N> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        let (idx, itm) = self.iter.next_back()?;
        let n = self.n.dim();
        Some((idx / n, idx % n, itm))
    }
}

impl<T, M: Dim, N: Dim> ExactSizeIterator for IterMut<'_, T, M, N> {}

impl<T, M: Dim, N: Dim> IntoIterator for Matrix<T, M, N> {
    type Item = (usize, usize, T);
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
    type Item = (usize, usize, &'a T);
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
    type Item = (usize, usize, &'a mut T);
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
}
