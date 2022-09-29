use crate::strategies::Actions;

pub trait ProdFunc {
    fn f_i(&self, i: usize, actions: &Actions) -> (f64, f64);
    fn f(&self, actions: &Actions) -> (Vec<f64>, Vec<f64>) {
        (0..actions.n).map(|i| self.f_i(i, actions)).unzip()
    }
}

pub struct DefaultProd {
    pub a: Vec<f64>,
    pub alpha: Vec<f64>,
    pub b: Vec<f64>,
    pub beta: Vec<f64>,
}

impl ProdFunc for DefaultProd {
    fn f_i(&self, i: usize, actions: &Actions) -> (f64, f64) {
        (
            self.a[i] * actions.xs[i].powf(self.alpha[i]),
            self.b[i] * actions.xp[i].powf(self.beta[i])
        )
    }
}
