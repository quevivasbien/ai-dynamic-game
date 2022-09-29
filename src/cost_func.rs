use crate::strategies::Actions;

pub trait CostFunc {
    fn c_i(&self, i: usize, actions: &Actions) -> f64;
    fn c(&self, actions: &Actions) -> Vec<f64> {
        (0..actions.n).map(|i| self.c_i(i, actions)).collect()
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
    fn c_i(&self, i: usize, actions: &Actions) -> f64 {
        self.r[i] * (actions.xs[i] + actions.xp[i])
    }
}
