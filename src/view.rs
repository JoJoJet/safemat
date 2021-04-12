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

    type Transpose: View<'a, T, N, M>;
    /// Gets an interface for accessing a transposed form of this matrix;
    /// does not actually rearrange any data.
    fn transpose_ref(self) -> Self::Transpose;
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

    type Transpose = Transpose<'a, T, Ms, Ns, Mo, No>;
    fn transpose_ref(self) -> Self::Transpose {
        Transpose { view: self }
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
pub struct Transpose<'a, T, Ms, Ns, Mo, No> {
    view: MatrixView<'a, T, Ms, Ns, Mo, No>,
}

impl<'a, T, Ms, Ns, Mo, No> View<'a, T, Ns, Ms> for Transpose<'a, T, Ms, Ns, Mo, No>
where
    Ms: Dim,
    Ns: Dim,
    Mo: Dim,
    No: Dim,
{
    #[inline]
    fn m(&self) -> Ns {
        self.view.n.identity()
    }
    #[inline]
    fn n(&self) -> Ms {
        self.view.m.identity()
    }

    type Iter = crate::iter::MatrixViewIter<'a, T, Ms, Ns, Mo, No>;
    #[inline]
    fn iter(&self) -> Self::Iter {
        self.view.iter()
    }

    type Transpose = MatrixView<'a, T, Ms, Ns, Mo, No>;
    fn transpose_ref(self) -> Self::Transpose {
        self.view
    }
}

impl<'a, T, Ms, Ns, Mo, No> Index<usize> for Transpose<'a, T, Ms, Ns, Mo, No>
where
    Ms: Dim,
    Ns: Dim,
    Mo: Dim,
    No: Dim,
{
    type Output = T;
    #[inline]
    fn index(&self, idx: usize) -> &T {
        let i = idx / self.view.m.dim();
        let j = idx % self.view.m.dim();
        &self.view[[j, i]]
    }
}
impl<'a, T, Ms, Ns, Mo, No> Index<[usize; 2]> for Transpose<'a, T, Ms, Ns, Mo, No>
where
    Ms: Dim,
    Ns: Dim,
    Mo: Dim,
    No: Dim,
{
    type Output = T;
    #[inline]
    fn index(&self, [i, j]: [usize; 2]) -> &T {
        &self.view[[j, i]]
    }
}

impl<'a, T, M: Dim, N: Dim, M2: Identity<M>, N2: Identity<N>> View<'a, T, M, N>
    for &'a Matrix<T, M2, N2>
{
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

    type Transpose = Transpose<'a, T, M, N, M2, N2>;
    #[inline]
    fn transpose_ref(self) -> Self::Transpose {
        self.as_view().transpose_ref()
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

impl<'a, T, M: Dim, N: Dim, M2: Identity<M>, N2: Identity<N>> IntoView<'a, T, M, N>
    for &'a Matrix<T, M2, N2>
{
    type View = Self;
    #[inline]
    fn into_view(self) -> Self::View {
        self
    }
}

pub trait RowView<'a, T: 'a, N: Dim>: View<'a, T, dim!(1), N> {}

impl<'a, T: 'a, N: Dim, V> RowView<'a, T, N> for V where V: View<'a, T, dim!(1), N> {}

pub type RowViewImpl<'a, T, Ns, Mo, No> = MatrixView<'a, T, dim!(1), Ns, Mo, No>;

pub trait IntoRowView<'a, T: 'a, N: Dim>: IntoView<'a, T, dim!(1), N> {}

impl<'a, T: 'a, N: Dim, I> IntoRowView<'a, T, N> for I where I: IntoView<'a, T, dim!(1), N> {}

impl<'a, T: 'a, M: Dim, V> ColumnView<'a, T, M> for V where V: View<'a, T, M, dim!(1)> {}

pub type ColumnViewImpl<'a, T, Ms, Mo, No> = MatrixView<'a, T, Ms, dim!(1), Mo, No>;

pub trait ColumnView<'a, T: 'a, M: Dim>: View<'a, T, M, dim!(1)> {}

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

    #[inline]
    pub fn as_view<M2: Dim, N2: Dim>(&self) -> MatrixView<'_, T, M2, N2, M, N>
    where
        M: Identity<M2>,
        N: Identity<N2>,
    {
        MatrixView {
            mat: self,
            m: self.m.identity(),
            n: self.n.identity(),
            i: 0,
            j: 0,
        }
    }

    pub fn row_at(&self, i: usize) -> RowViewImpl<'_, T, N, M, N> {
        assert!(i < self.m.dim());
        MatrixView {
            mat: self,
            m: dim!(1),
            n: self.n,
            i,
            j: 0,
        }
    }

    pub fn col_at(&self, j: usize) -> ColumnViewImpl<'_, T, M, M, N> {
        assert!(j < self.n.dim());
        MatrixView {
            mat: self,
            m: self.m,
            n: dim!(1),
            i: 0,
            j,
        }
    }
}
