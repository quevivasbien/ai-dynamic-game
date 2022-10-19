use numpy::ndarray::{Array, Ix1};

use crate::strategies::*;

pub trait CostFunc: Clone + Send + Sync {
    type Act: ActionType;

    fn c_i(&self, i: usize, actions: &Self::Act) -> f64;
    fn c(&self, actions: &Self::Act) -> Array<f64, Ix1> {
        Array::from_iter((0..actions.n()).map(|i| self.c_i(i, actions)))
    }

    fn n(&self) -> usize;
}

#[derive(Clone)]
pub struct FixedUnitCost {
    pub r: Array<f64, Ix1>,
}

impl FixedUnitCost {
    pub fn from_elem(n: usize, r: f64) -> FixedUnitCost {
        FixedUnitCost {
            r: Array::from_elem(n, r)
        }
    }
}

impl CostFunc for FixedUnitCost {
    type Act = Actions;

    fn c_i(&self, i: usize, actions: &Actions) -> f64 {
        self.r[i] * (actions.xs()[i] + actions.xp()[i])
    }
    
    fn n(&self) -> usize {
        self.r.len()
    }
}

#[derive(Clone)]
pub struct FixedInvestCost {
    n: usize,
    r_x: Array<f64, Ix1>,
    r_inv: Array<f64, Ix1>,
}

impl FixedInvestCost {
    pub fn new(r_x: Array<f64, Ix1>, r_inv: Array<f64, Ix1>) -> FixedInvestCost {
        assert_eq!(r_x.len(), r_inv.len());
        FixedInvestCost {
            n: r_x.len(),
            r_x,
            r_inv,
        }
    }
    pub fn from_elems(n: usize, r_x: f64, r_inv: f64) -> FixedInvestCost {
        FixedInvestCost {
            n,
            r_x: Array::from_elem(n, r_x),
            r_inv: Array::from_elem(n, r_inv),
        }
    }
}

impl CostFunc for FixedInvestCost {
    type Act = InvestActions;

    fn c_i(&self, i: usize, actions: &InvestActions) -> f64 {
        self.r_x[i] * (actions.xs()[i] + actions.xp()[i]) + self.r_inv[i] * (actions.inv_s()[i] + actions.inv_p()[i])
    }

    fn n(&self) -> usize {
        self.n
    }
}