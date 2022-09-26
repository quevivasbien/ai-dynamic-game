pub trait RewardFunc {
    fn win_i(&self, i: usize, p: &Vec<f64>) -> f64;
    fn lose_i(&self, i: usize, p: &Vec<f64>) -> f64;
    fn reward(&self, i: usize, p: &Vec<f64>) -> Vec<f64> {
        (0..p.len()).map(|j| {
            if j == i {
                self.win_i(j, p)
            } else {
                self.lose_i(j, p)
            }
        }).collect()
    }
}

pub struct LinearReward {
    win_a: Vec<f64>,
    win_b: Vec<f64>,
    lose_a: Vec<f64>,
    lose_b: Vec<f64>,
}

impl LinearReward {
    pub fn new(n: usize) -> LinearReward {
        LinearReward {
            win_a: vec![1.0; n],
            win_b: vec![0.0; n],
            lose_a: vec![0.0; n],
            lose_b: vec![0.0; n],
        }
    }
}

impl RewardFunc for LinearReward {
    fn win_i(&self, i: usize, p: &Vec<f64>) -> f64 {
        self.win_a[i] + self.win_b[i] * p[i]
    }
    fn lose_i(&self, i: usize, p: &Vec<f64>) -> f64 {
        self.lose_a[i] + self.lose_b[i] * p[i]
    }
}
