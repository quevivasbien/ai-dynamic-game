use numpy::ndarray::{ArrayView, Ix1, Array};

pub trait RiskFunc {
    // sigma_i is proba(safe | i wins)
    fn sigma_i(&self, i: usize, s: ArrayView<f64, Ix1>, p: ArrayView<f64, Ix1>) -> f64;
    fn sigma(&self, s: ArrayView<f64, Ix1>, p: ArrayView<f64, Ix1>) -> Array<f64, Ix1> {
        Array::from_iter((0..s.len()).map(|i| self.sigma_i(i, s, p)))
    }
}

#[derive(Clone)]
pub struct WinnerOnlyRisk {
    pub theta: Array<f64, Ix1>,
}

impl RiskFunc for WinnerOnlyRisk {
    fn sigma_i(&self, i: usize, s: ArrayView<f64, Ix1>, p: ArrayView<f64, Ix1>) -> f64 {
        let s_ = s[i] * p[i].powf(-self.theta[i]);
        s_ / (1.0 + s_)
    }
}

impl WinnerOnlyRisk {
    pub fn new(n: usize, theta: f64) -> Self {
        WinnerOnlyRisk {
            theta: Array::from_elem(n, theta),
        }
    }
}