use numpy::ndarray::{Array, Ix1};

use crate::strategies::*;
use crate::payoff_func::PayoffFunc;

pub trait State<T: PayoffFunc>: Clone {
    fn belief(&self, i: usize) -> &T;
}

impl<T: PayoffFunc + Clone> State<T> for T {
    fn belief(&self, _i: usize) -> &Self {
        self
    }
}

#[derive(Clone)]
pub struct HetBeliefs<T: PayoffFunc> {
    pub beliefs: Vec<T>,
}

impl<T: PayoffFunc + Clone> State<T> for HetBeliefs<T> {
    fn belief(&self, i: usize) -> &T {
        &self.beliefs[i]
    }
}

pub trait PayoffAggregator {
    type Strat: StrategyType;
    fn u_i(&self, i: usize, strategies: &Self::Strat) -> f64;
    fn u(&self, strategies: &Self::Strat) -> Array<f64, Ix1> {
        Array::from_iter((0..strategies.n()).map(|i| self.u_i(i, strategies)))
    }
}

pub struct ExponentialDiscounter<U: PayoffFunc, T: State<U>> {
    pub states: Vec<T>,
    pub gammas: Vec<f64>,
    phantom: std::marker::PhantomData<U>,
}

impl<U: PayoffFunc, T: State<U>> ExponentialDiscounter<U, T> {
    pub fn new(states: Vec<T>, gammas: Vec<f64>) -> Self {
        ExponentialDiscounter {
            states,
            gammas,
            phantom: std::marker::PhantomData,
        }
    }
    pub fn new_static(state0: T, t: usize, gammas: Vec<f64>) -> Self {
        Self::new(vec![state0; t], gammas)
    }
}

impl<U: PayoffFunc, T: State<U>> PayoffAggregator for ExponentialDiscounter<U, T> {
    type Strat = Strategies;

    fn u_i(&self, i: usize, strategies: &Strategies) -> f64 {
        assert_eq!(self.states.len(), strategies.t, "Number of states should match number of time periods in strategies");
        let actions_seq = strategies.clone().to_actions();
        actions_seq.iter().enumerate().map(|(t, actions)| {
            let gamma = self.gammas[i];
            let belief = self.states[t].belief(i);
            gamma.powi(t.try_into().unwrap()) * belief.u_i(i, actions)
        }).sum()
    }
}
