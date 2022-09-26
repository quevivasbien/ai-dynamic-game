pub trait ProdFunc {
    fn f_i(&self, i: usize, xs: &Vec<f64>, xp: &Vec<f64>) -> (f64, f64);
    fn f(&self, xs: &Vec<f64>, xp: &Vec<f64>) -> (Vec<f64>, Vec<f64>) {
        (0..xs.len()).map(|i| self.f_i(i, xs, xp)).unzip()
    }
}

pub struct DefaultProd {
    pub a: Vec<f64>,
    pub alpha: Vec<f64>,
    pub b: Vec<f64>,
    pub beta: Vec<f64>,
}

impl ProdFunc for DefaultProd {
    fn f_i(&self, i: usize, xs: &Vec<f64>, xp: &Vec<f64>) -> (f64, f64) {
        (
            self.a[i] * xs[i].powf(self.alpha[i]),
            self.b[i] * xp[i].powf(self.beta[i])
        )
    }
}
