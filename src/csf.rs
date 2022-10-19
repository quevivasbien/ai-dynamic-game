use numpy::ndarray::{Array, ArrayView, Ix1};

pub trait CSF: Clone + Send + Sync {
    fn q_i(&self, i: usize, p: ArrayView<f64, Ix1>) -> f64;
    fn q(&self, p: ArrayView<f64, Ix1>) -> Array<f64, Ix1> {
        Array::from_iter((0..p.len()).map(|i| self.q_i(i, p)))
    }
}

#[derive(Clone, Debug)]
pub struct DefaultCSF;

impl CSF for DefaultCSF {
    fn q_i(&self, i: usize, p: ArrayView<f64, Ix1>) -> f64 {
        let sum_p: f64 = p.iter().sum();
        if sum_p == 0.0 {
            0.0
        } else {
            p[i] / sum_p
        }
    }

    fn q(&self, p: ArrayView<f64, Ix1>) -> Array<f64, Ix1> {
        let sum_p: f64 = p.iter().sum();
        if sum_p == 0.0 {
            Array::zeros(p.len())
        } else {
            Array::from_iter(p.iter().map(|x| x / sum_p))
        }
    }
}
