use numpy::ndarray::{Array, ArrayView, Ix1};

pub trait RewardFunc {
    fn win_i(&self, i: usize, p: ArrayView<f64, Ix1>) -> f64;
    fn lose_i(&self, i: usize, p: ArrayView<f64, Ix1>) -> f64;
    fn reward(&self, i: usize, p: ArrayView<f64, Ix1>) -> Array<f64, Ix1> {
        Array::from_iter((0..p.len()).map(|j| {
            if j == i {
                self.win_i(j, p)
            } else {
                self.lose_i(j, p)
            }
        }))
    }
}

#[derive(Clone)]
pub struct LinearReward {
    pub win_a: Array<f64, Ix1>,
    pub win_b: Array<f64, Ix1>,
    pub lose_a: Array<f64, Ix1>,
    pub lose_b: Array<f64, Ix1>,
}

impl LinearReward {
    pub fn default(n: usize) -> Self {
        LinearReward {
            win_a: Array::ones(n),
            win_b: Array::zeros(n),
            lose_a: Array::zeros(n),
            lose_b: Array::zeros(n),
        }
    }
}

impl RewardFunc for LinearReward {
    fn win_i(&self, i: usize, p: ArrayView<f64, Ix1>) -> f64 {
        self.win_a[i] + self.win_b[i] * p[i]
    }
    fn lose_i(&self, i: usize, p: ArrayView<f64, Ix1>) -> f64 {
        self.lose_a[i] + self.lose_b[i] * p[i]
    }
}
