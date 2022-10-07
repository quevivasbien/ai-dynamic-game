use numpy::ndarray::{Array, Ix1};

use crate::strategies::Actions;

pub trait CostFunc {
    fn c_i(&self, i: usize, actions: &Actions) -> f64;
    fn c(&self, actions: &Actions) -> Array<f64, Ix1> {
        Array::from_iter((0..actions.n).map(|i| self.c_i(i, actions)))
    }
}

#[derive(Clone)]
pub struct FixedUnitCost {
    r: Array<f64, Ix1>,
}

impl FixedUnitCost {
    pub fn new(n: usize, r: f64) -> FixedUnitCost {
        FixedUnitCost {
            r: Array::from_elem(n, r)
        }
    }
}

impl CostFunc for FixedUnitCost {
    fn c_i(&self, i: usize, actions: &Actions) -> f64 {
        self.r[i] * (actions.xs()[i] + actions.xp()[i])
    }
}
