pub trait CostFunc {
    fn c_i(&self, i: usize, xs: &Vec<f64>, xp: &Vec<f64>) -> f64;
    fn c(&self, xs: &Vec<f64>, xp: &Vec<f64>) -> Vec<f64> {
        (0..xs.len()).map(|i| self.c_i(i, xs, xp)).collect()
    }
}

pub struct DefaultCost {
    r: Vec<f64>,
}

impl DefaultCost {
    pub fn new(n: usize, r: f64) -> DefaultCost {
        DefaultCost {
            r: vec![r; n],
        }
    }
}

impl CostFunc for DefaultCost {
    fn c_i(&self, i: usize, xs: &Vec<f64>, xp: &Vec<f64>) -> f64 {
        self.r[i] * (xs[i] + xp[i])
    }
}
