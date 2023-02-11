use super::{DesignMatrix, DesignMatrixType};
use crate::Float;
use ndarray::{s, Array1, Array2, ArrayBase, ArrayView1, Axis, Data, Ix2};
use sprs::CsMat;

/// This implements the [`DesignMatrix`] trait for dense matrices.
impl<F: Float, S: Data<Elem = F>> DesignMatrix for ArrayBase<S, Ix2> {
    type Elem = F;

    fn n_features(&self) -> usize {
        self.len_of(Axis(1))
    }

    fn matrix_type(&self) -> DesignMatrixType {
        DesignMatrixType::Dense
    }

    fn compute_extrapolated_fit(
        &self,
        ws: ArrayView1<usize>,
        w_acc: &Array1<F>,
        _n_samples: usize,
    ) -> Array1<F> {
        Array1::from_iter(
            self.rows()
                .into_iter()
                .map(|row| ws.iter().map(|&j| row[j] * w_acc[j]).sum())
                .collect::<Vec<F>>(),
        )
    }

    fn compute_extrapolated_fit_multi_task(
        &self,
        ws: ArrayView1<usize>,
        W_acc: &Array2<F>,
        n_samples: usize,
        n_tasks: usize,
    ) -> Array2<F> {
        Array2::from_shape_vec(
            (n_samples, n_tasks),
            self.rows()
                .into_iter()
                .map(|row| {
                    W_acc
                        .columns()
                        .into_iter()
                        .map(|col| ws.iter().map(|&j| row[j] * col[j]).sum())
                        .collect::<Vec<F>>()
                })
                .collect::<Vec<Vec<F>>>()
                .into_iter()
                .flatten()
                .collect::<Vec<F>>(),
        )
        .unwrap()
    }

    fn update_model_fit(&self, Xw: &mut Array1<Self::Elem>, diff: Self::Elem, j: usize) {
        Xw.scaled_add(diff, &self.slice(s![.., j]));
    }

    fn update_model_fit_multi_task(
        &self,
        XW: &mut Array2<Self::Elem>,
        diff: ArrayView1<Self::Elem>,
        j: usize,
    ) {
        let n_samples = XW.shape()[0];
        let n_tasks = XW.shape()[1];
        for i in 0..n_samples {
            for t in 0..n_tasks {
                XW[[i, t]] += diff[t] * self[[i, j]];
            }
        }
    }
}

/// This implements the [`DesignMatrix`] trait for sparse matrices.
impl<F: Float> DesignMatrix for CSCArray<'_, F> {
    type Elem = F;

    fn n_features(&self) -> usize {
        self.indptr.len() - 1
    }

    fn matrix_type(&self) -> DesignMatrixType {
        DesignMatrixType::Sparse
    }

    fn compute_extrapolated_fit(
        &self,
        ws: ArrayView1<usize>,
        w_acc: &Array1<F>,
        n_samples: usize,
    ) -> Array1<F> {
        let mut Xw_acc = Array1::<F>::zeros(n_samples);
        for &j in ws {
            for idx in self.indptr[j]..self.indptr[j + 1] {
                Xw_acc[self.indices[idx as usize] as usize] += self.data[idx as usize] * w_acc[j];
            }
        }
        Xw_acc
    }

    fn compute_extrapolated_fit_multi_task(
        &self,
        ws: ArrayView1<usize>,
        W_acc: &Array2<F>,
        n_samples: usize,
        n_tasks: usize,
    ) -> Array2<F> {
        let mut XW_acc = Array2::<F>::zeros((n_samples, n_tasks));
        for &j in ws {
            for idx in self.indptr[j]..self.indptr[j + 1] {
                for t in 0..n_tasks {
                    XW_acc[[self.indices[idx as usize] as usize, t]] +=
                        self.data[idx as usize] * W_acc[[j, t]];
                }
            }
        }
        XW_acc
    }

    fn update_model_fit(&self, Xw: &mut Array1<Self::Elem>, diff: Self::Elem, j: usize) {
        for i in self.indptr[j]..self.indptr[j + 1] {
            Xw[self.indices[i as usize] as usize] += diff * self.data[i as usize];
        }
    }

    fn update_model_fit_multi_task(
        &self,
        XW: &mut Array2<Self::Elem>,
        diff: ArrayView1<Self::Elem>,
        j: usize,
    ) {
        let n_tasks = XW.shape()[1];
        for i in self.indptr[j]..self.indptr[j + 1] {
            for t in 0..n_tasks {
                XW[[self.indices[i as usize] as usize, t]] += diff[t] * self.data[i as usize];
            }
        }
    }
}

/// This implements the [`DesignMatrix`] trait for references.
impl<DM: DesignMatrix> DesignMatrix for &DM {
    type Elem = DM::Elem;

    fn n_features(&self) -> usize {
        (*self).n_features()
    }

    fn matrix_type(&self) -> DesignMatrixType {
        (*self).matrix_type()
    }

    fn compute_extrapolated_fit(
        &self,
        ws: ArrayView1<usize>,
        w_acc: &Array1<Self::Elem>,
        n_samples: usize,
    ) -> Array1<Self::Elem> {
        (*self).compute_extrapolated_fit(ws, w_acc, n_samples)
    }

    fn compute_extrapolated_fit_multi_task(
        &self,
        ws: ArrayView1<usize>,
        W_acc: &Array2<Self::Elem>,
        n_samples: usize,
        n_tasks: usize,
    ) -> Array2<Self::Elem> {
        (*self).compute_extrapolated_fit_multi_task(ws, W_acc, n_samples, n_tasks)
    }

    fn update_model_fit(&self, Xw: &mut Array1<Self::Elem>, diff: Self::Elem, j: usize) {
        (*self).update_model_fit(Xw, diff, j);
    }

    fn update_model_fit_multi_task(
        &self,
        XW: &mut Array2<Self::Elem>,
        diff: ArrayView1<Self::Elem>,
        j: usize,
    ) {
        (*self).update_model_fit_multi_task(XW, diff, j);
    }
}
