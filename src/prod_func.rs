use numpy::ndarray::{Array, Ix1};

use crate::strategies::Actions;

pub trait ProdFunc {
    fn f_i(&self, i: usize, actions: &Actions) -> (f64, f64);
    fn f(&self, actions: &Actions) -> (Array<f64, Ix1>, Array<f64, Ix1>) {
        let (s, p) = (0..actions.n).map(|i| self.f_i(i, actions)).unzip();
        (Array::from_vec(s), Array::from_vec(p))
    }
}

#[derive(Clone)]
pub struct DefaultProd {
    pub a: Array<f64, Ix1>,
    pub alpha: Array<f64, Ix1>,
    pub b: Array<f64, Ix1>,
    pub beta: Array<f64, Ix1>,
}

impl ProdFunc for DefaultProd {
    fn f_i(&self, i: usize, actions: &Actions) -> (f64, f64) {
        (
            self.a[i] * actions.xs()[i].powf(self.alpha[i]),
            self.b[i] * actions.xp()[i].powf(self.beta[i])
        )
    }
}
