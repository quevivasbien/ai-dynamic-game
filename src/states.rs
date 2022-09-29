use std::rc::Rc;

use crate::strategies::*;
use crate::payoff_func::PayoffFunc;

pub trait State {
    fn belief(&self, i: usize) -> &Box<dyn PayoffFunc>;
}

pub struct CommonBeliefs {
    pub belief: Box<dyn PayoffFunc>,
}

impl State for CommonBeliefs {
    fn belief(&self, _i: usize) -> &Box<dyn PayoffFunc> {
        &self.belief
    }
}

pub struct HetBeliefs {
    pub beliefs: Vec<Box<dyn PayoffFunc>>,
}

impl State for HetBeliefs {
    fn belief(&self, i: usize) -> &Box<dyn PayoffFunc> {
        &self.beliefs[i]
    }
}

pub trait PayoffAggregator {
    fn u_i(&self, i: usize, strategies: &Strategies) -> f64;
    fn u(&self, strategies: &Strategies) -> Vec<f64> {
        (0..strategies.n).map(|i| self.u_i(i, strategies)).collect()
    }
}

pub struct ExponentialDiscounter<T: State> {
    pub states: Vec<Rc<T>>,
    pub gammas: Vec<f64>,
}

impl<T: State> PayoffAggregator for ExponentialDiscounter<T> {
    fn u_i(&self, i: usize, strategies: &Strategies) -> f64 {
        self.states.iter().enumerate().map(|(t, state)| {
            self.gammas[i].powi(t.try_into().unwrap()) * state.belief(i).u_i(i, &strategies.x[t])
        }).sum()
    }
}
