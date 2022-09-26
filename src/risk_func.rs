pub trait RiskFunc {
    // sigma_i is proba(safe | i wins)
    fn sigma_i(&self, i: usize, s: &Vec<f64>, p: &Vec<f64>) -> f64;
    fn sigma(&self, s: &Vec<f64>, p: &Vec<f64>) -> Vec<f64> {
        (0..s.len()).map(|i| self.sigma_i(i, s, p)).collect()
    }
}

pub struct WinnerOnlyRisk {
    pub theta: Vec<f64>,
}

impl RiskFunc for WinnerOnlyRisk {
    fn sigma_i(&self, i: usize, s: &Vec<f64>, p: &Vec<f64>) -> f64 {
        let s_ = s[i] * p[i].powf(-self.theta[i]);
        s_ / (1.0 + s_)
    }
}