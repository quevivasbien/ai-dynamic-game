use std::rc::Rc;

use ndarray::{Array, Ix1};

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
    fn u(&self, strategies: &Strategies) -> Array<f64, Ix1> {
        Array::from_iter((0..strategies.n).map(|i| self.u_i(i, strategies)))
    }
}

pub struct ExponentialDiscounter<T: State> {
    pub states: Vec<Rc<T>>,
    pub gammas: Vec<f64>,
}

impl<T: State> PayoffAggregator for ExponentialDiscounter<T> {
    fn u_i(&self, i: usize, strategies: &Strategies) -> f64 {
        assert_eq!(self.states.len(), strategies.t);
        let actions_seq = strategies.clone().to_actions();
        actions_seq.iter().enumerate().map(|(t, actions)| {
            let gamma = self.gammas[i];
            let belief = self.states[t].belief(i);
            gamma.powi(t.try_into().unwrap()) * belief.u_i(i, actions)
        }).sum()
    }
}
