pub trait DisasterCost {
    fn d_i(&self, i: usize, s: &Vec<f64>, p: &Vec<f64>) -> f64;
    fn d(&self, s: &Vec<f64>, p: &Vec<f64>) -> Vec<f64> {
        (0..s.len()).map(|i| self.d_i(i, s, p)).collect()
    }
}

pub struct ConstantDisasterCost {
    pub d: Vec<f64>,
}

impl DisasterCost for ConstantDisasterCost {
    fn d_i(&self, i: usize, _s: &Vec<f64>, _p: &Vec<f64>) -> f64 {
        self.d[i]
    }
}
