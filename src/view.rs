use std::{marker::PhantomData, ops::Index};

use crate::{Dim, Matrix};

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

    /// Gets the entry at specified indices,
    /// or returns nothing if the index is invalid.
    fn get(&self, i: usize, j: usize) -> Option<&'a Self::Entry>;

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
    /// # Examples
    /// Row/column slices.
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
    /// # Examples
    /// ```
    /// # use safemat::prelude::*;
    /// let a = mat![1, 2, 3 ; 4, 5, 6];
    /// let b = a.as_view().transpose_ref().to_matrix();
    /// assert_eq!(b, mat![1, 4 ; 2, 5 ; 3, 6]);
    /// ```
    /// It also supports iteration that yields indices.
    /// ```
    /// # use safemat::prelude::*;
    /// let a = Matrix::from_array([
    ///     [1, 2, 3],
    ///     [4, 5, 6],
    /// ]);
    /// let b = a.as_view().transpose_ref();
    /// let mut i = b.iter().indices();
    /// assert_eq!(i.next(), Some((0, 0, &1)));
    /// assert_eq!(i.next(), Some((0, 1, &4)));
    /// assert_eq!(i.next(), Some((1, 0, &2)));
    /// // etc...
    /// # assert_eq!(i.next(), Some((1, 1, &5)));
    /// # assert_eq!(i.next(), Some((2, 0, &3)));
    /// # assert_eq!(i.next(), Some((2, 1, &6)));
    /// ```
    #[inline]
    fn transpose_ref(self) -> Transpose<'a, Self> {
        Transpose {
            view: self,
            _p: PhantomData,
        }
    }
}

pub mod row;

pub mod col;

impl<T, M: Dim, N: Dim> Matrix<T, M, N> {
    #[inline]
    pub fn as_view(&self) -> &Self {
        self
    }

    pub fn row_at(&self, i: usize) -> row::RowSlice<'_, T, M, N> {
        row::RowSlice::new(self, i)
    }

    #[inline]
    pub fn col_at(&self, j: usize) -> col::ColumnSlice<'_, T, M, N> {
        col::ColumnSlice::new(self, j)
    }
}

/// Interchanges the indices when accessing a matrix view.
/// ```
/// # use safemat::prelude::*;
/// let a = Matrix::from_array([
///     [1, 2, 3, 4],
///     [5, 6, 7, 8],
/// ]);
/// let t = a.as_view().transpose_ref();
/// // linear indexing:
/// assert_eq!(t[0], 1);
/// assert_eq!(t[1], 5);
/// assert_eq!(t[2], 2);
/// // etc...
/// # assert_eq!(t[3], 6);
/// # assert_eq!(t[4], 3);
/// # assert_eq!(t[5], 7);
/// # assert_eq!(t[6], 4);
/// # assert_eq!(t[7], 8);
///
/// // square indexing:
/// assert_eq!(t[[0,0]], 1);
/// assert_eq!(t[[1,0]], 2);
/// assert_eq!(t[[2,0]], 3);
/// // etc...
/// # assert_eq!(t[[3,0]], 4);
/// # assert_eq!(t[[0,1]], 5);
/// # assert_eq!(t[[1,1]], 6);
/// # assert_eq!(t[[2,1]], 7);
/// # assert_eq!(t[[3,1]], 8);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Transpose<'a, V: 'a> {
    view: V,
    _p: PhantomData<&'a V>,
}

impl<'a, V> View<'a> for Transpose<'a, V>
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

    #[inline]
    fn get(&self, i: usize, j: usize) -> Option<&'a Self::Entry> {
        self.view.get(j, i)
    }

    type Iter = TransposeIter<'a, V>;
    #[inline]
    fn iter(&self) -> Self::Iter {
        TransposeIter {
            view: self.view,
            i: 0,
            j: 0,
            _p: PhantomData,
        }
    }
}

impl<'a, V> Index<usize> for Transpose<'a, V>
where
    V: View<'a>,
{
    type Output = V::Entry;
    #[inline]
    fn index(&self, idx: usize) -> &Self::Output {
        let i = idx / self.view.m().dim();
        let j = idx % self.view.m().dim();
        &self.view[[j, i]]
    }
}
impl<'a, V> Index<[usize; 2]> for Transpose<'a, V>
where
    V: View<'a>,
{
    type Output = V::Entry;
    #[inline]
    fn index(&self, [i, j]: [usize; 2]) -> &Self::Output {
        &self.view[[j, i]]
    }
}

/// An iterator over the transpose of a [`View`].
pub struct TransposeIter<'a, V> {
    view: V,
    _p: PhantomData<&'a V>,
    i: usize,
    j: usize,
}

impl<'a, V: View<'a>> TransposeIter<'a, V> {
    #[inline]
    pub fn indices(self) -> crate::iter::Indices<Self> {
        crate::iter::Indices(self)
    }
}

impl<'a, V> Iterator for TransposeIter<'a, V>
where
    V: View<'a>,
{
    type Item = &'a V::Entry;
    fn next(&mut self) -> Option<Self::Item> {
        let j = self.j;
        if j < self.view.n().dim() {
            let i = self.i;
            self.i += 1;
            if self.i == self.view.m().dim() {
                self.i = 0;
                self.j += 1;
            }
            self.view.get(i, j) // Due to this line, this is NOT a fused iterator.
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

impl<'a, V: View<'a>> crate::iter::IndexIterator for TransposeIter<'a, V> {
    fn get_indices(&self) -> (usize, usize) {
        (self.j, self.i)
    }

    fn get_indices_back(&self) -> (usize, usize)
    where
        Self: DoubleEndedIterator,
    {
        unimplemented!() // we don't support this 'round these parts
    }
}

impl<'a, V> ExactSizeIterator for TransposeIter<'a, V>
where
    V: View<'a>,
{
    #[inline]
    fn len(&self) -> usize {
        let idx = self.i * self.view.n().dim() + self.j;
        self.view.m().dim() * self.view.n().dim() - idx
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

    #[inline]
    fn get(&self, i: usize, j: usize) -> Option<&'a T> {
        (*self).get(i, j)
    }

    type Iter = crate::iter::Iter<'a, T, M, N>;
    #[inline]
    fn iter(&self) -> Self::Iter {
        Matrix::iter(self)
    }
}

pub mod entry {
    use std::ops::Index;

    use super::View;
    use crate::{dim, dim::Dim, Matrix};

    
    /// A [`View`] referring to a single entry in a matrix.
    /// # Examples
    /// ```
    /// # use safemat::{prelude::*, view::entry::Entry};
    /// let m = mat![1, 2; 3, 4];
    /// assert_eq!(Entry::new(&m, 0, 1), &mat![2]);
    /// assert_eq!(Entry::new(&m, 1, 0), &mat![3]);
    #[derive(Debug)]
    pub struct Entry<'a, T, M, N> {
        mat: &'a Matrix<T, M, N>,
        i: usize,
        j: usize,
    }

    impl<'a, T, M: Dim, N: Dim> Entry<'a, T, M, N> {
        #[inline]
        pub fn new(mat: &'a Matrix<T, M, N>, i: usize, j: usize) -> Self {
            assert!(i < mat.m.dim());
            assert!(j < mat.n.dim());
            Self { mat, i, j }
        }
    }

    impl<T, M, N> Copy for Entry<'_, T, M, N> {}

    impl<T, M, N> Clone for Entry<'_, T, M, N> {
        fn clone(&self) -> Self {
            Self {
                mat: self.mat,
                i: self.i,
                j: self.j,
            }
        }
    }

    impl<'a, T, M: Dim, N: Dim> View<'a> for Entry<'a, T, M, N> {
        type Entry = T;
        type M = dim!(1);
        type N = dim!(1);
        fn m(&self) -> dim!(1) {
            dim!(1)
        }
        fn n(&self) -> dim!(1) {
            dim!(1)
        }

        fn get(&self, i: usize, j: usize) -> Option<&'a T> {
            if i == 0 && j == 0 {
                Some(&self.mat.items[self.i * self.mat.n.dim() + self.j])
            } else {
                None
            }
        }

        type Iter = std::iter::Once<&'a T>;
        fn iter(&self) -> Self::Iter {
            std::iter::once(self.get(0, 0).unwrap())
        }
    }

    impl<T, M: Dim, N: Dim> Index<usize> for Entry<'_, T, M, N> {
        type Output = T;
        fn index(&self, i: usize) -> &T {
            assert_eq!(i, 0);
            &self.mat[[self.i, self.j]]
        }
    }
    impl<T, M: Dim, N: Dim> Index<[usize; 2]> for Entry<'_, T, M, N> {
        type Output = T;
        fn index(&self, [i, j]: [usize; 2]) -> &T {
            assert_eq!(i, 0);
            assert_eq!(j, 0);
            &self.mat[[self.i, self.j]]
        }
    }

    impl<'a, T, M: Dim, N: Dim, V> PartialEq<V> for Entry<'a, T, M, N>
    where
        V: View<'a>,
        T: PartialEq<V::Entry>,
        dim!(1): PartialEq<V::M> + PartialEq<V::N>,
    {
        #[inline]
        fn eq(&self, rhs: &V) -> bool {
            self.equal(*rhs)
        }
    }
}
