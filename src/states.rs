use std::marker::PhantomData;

use numpy::ndarray::{Array, Ix1};

use crate::strategies::*;
use crate::payoff_func::PayoffFunc;

pub trait State<T: PayoffFunc>: Clone + Send + Sync {
    fn n(&self) -> usize;
    fn belief(&self, i: usize) -> &T;
}

impl<T: PayoffFunc> State<T> for T {
    fn n(&self) -> usize {
        self.n()
    }
    fn belief(&self, _i: usize) -> &Self {
        self
    }
}

#[derive(Clone)]
pub struct HetBeliefs<T: PayoffFunc> {
    n: usize,
    beliefs: Vec<T>,
}

impl<T: PayoffFunc> State<T> for HetBeliefs<T> {
    fn n(&self) -> usize {
        self.n
    }
    fn belief(&self, i: usize) -> &T {
        &self.beliefs[i]
    }
}

impl<T: PayoffFunc> HetBeliefs<T> {
    pub fn new(beliefs: Vec<T>) -> Result<HetBeliefs<T>, &'static str> {
        if beliefs.len() == 0 {
            return Err("When creating new HetBeliefs: beliefs must have length > 0");
        }
        let n = beliefs[0].n();
        if beliefs.iter().any(|b| b.n() != n) {
            return Err("When creating new HetBeliefs: All beliefs must have the same n");
        }
        Ok(HetBeliefs { n, beliefs })
    }
}


pub trait StateIterator<A, S>: Send + Sync
where A: ActionType, S: StrategyType<Act = A>
{
    type PFunc: PayoffFunc<Act = A>;
    type StateType: State<Self::PFunc>;
    fn state0(&self) -> &Self::StateType;
    fn advance_state(&self, _state: &mut Self::StateType, _actions: &A) {}
}

pub trait PayoffAggregator<A, S>: Send + Sync
where A: ActionType, S: StrategyType<Act = A>
{
    fn n(&self) -> usize;
    fn u_i(&self, i: usize, strategies: &S) -> f64;
    fn u(&self, strategies: &S) -> Array<f64, Ix1> {
        Array::from_iter((0..strategies.n()).map(|i| self.u_i(i, strategies)))
    }
}

pub trait Discounter {
    fn gammas(&self) -> &Array<f64, Ix1>;
}

impl<A, S, T> PayoffAggregator<A, S> for T
where A: ActionType, S: StrategyType<Act = A>, T: StateIterator<A, S> + Discounter
{
    fn n(&self) -> usize {
        self.state0().n()
    }
    fn u_i(&self, i: usize, strategies: &S) -> f64 {
        let actions_seq = strategies.clone().to_actions();
        let state = &mut self.state0().clone();
        let gammas = self.gammas();
        let mut u = 0.0;
        for (t, actions) in actions_seq.iter().enumerate() {
            u += gammas[i].powi(t.try_into().unwrap()) * state.belief(i).u_i(i, actions);
            if t != actions_seq.len() - 1 {
                self.advance_state(state, actions);
            }
        }
        u
    }
    fn u(&self, strategies: &S) -> Array<f64, Ix1> {
        let actions_seq = strategies.clone().to_actions();
        let state = &mut self.state0().clone();
        let gammas = self.gammas();
        let mut u: Array<f64, Ix1> = Array::zeros(gammas.len());
        for (t, actions) in actions_seq.iter().enumerate() {
            u.iter_mut().zip(gammas.iter()).enumerate().for_each(|(i, (u_i, gamma))| {
                *u_i += gamma.powi(t.try_into().unwrap()) * state.belief(i).u_i(i, actions);
            });
            if t != actions_seq.len() - 1 {
                self.advance_state(state, actions);
            }
        }
        u
    }
}

#[derive(Clone)]
pub struct FixedStateDiscounter<A, S, P, T>
where A: ActionType, S: StrategyType<Act = A>, P: PayoffFunc<Act = A>, T: State<P>
{
    state: T,
    gammas: Array<f64, Ix1>,
    _phantoms: PhantomData<(A, S, P)>,
}

impl<A, S, P, T> FixedStateDiscounter<A, S, P, T>
where A: ActionType, S: StrategyType<Act = A>, P: PayoffFunc<Act = A>, T: State<P>
{
    pub fn new(state: T, gammas: Array<f64, Ix1>) -> Result<Self, &'static str> {
        if state.n() != gammas.len() {
            return Err("When creating new FixedStateDiscounter: gammas must have length == n");
        }
        Ok(FixedStateDiscounter { state, gammas, _phantoms: PhantomData })
    }
}

impl<A, S, P, T> StateIterator<A, S> for FixedStateDiscounter<A, S, P, T>
where A: ActionType, S: StrategyType<Act = A>, P: PayoffFunc<Act = A>, T: State<P>
{
    type PFunc = P;
    type StateType = T;
    fn state0(&self) -> &T {
        &self.state
    }
}

impl<A, S, P, T> Discounter for FixedStateDiscounter<A, S, P, T>
where A: ActionType, S: StrategyType<Act = A>, P: PayoffFunc<Act = A>, T: State<P>
{
    fn gammas(&self) -> &Array<f64, Ix1> {
        &self.gammas
    }
}

pub type ExponentialDiscounter<P, T> = FixedStateDiscounter<Actions, Strategies, P, T>;

#[derive(Clone)]
pub struct DynStateDiscounter<A, S, P, T>
where A: ActionType,
      S: StrategyType<Act = A>,
      P: PayoffFunc<Act = A>, T: State<P>,
      T: State<P> + MutatesOnAction<A>,
{
    state0: T,
    gammas: Array<f64, Ix1>,
    _phantoms: PhantomData<(A, S, P)>,
}

impl<A, S, P, T> DynStateDiscounter<A, S, P, T>
where A: ActionType,
      S: StrategyType<Act = A>,
      P: PayoffFunc<Act = A>, T: State<P>,
      T: State<P> + MutatesOnAction<A>,
{
    pub fn new(state0: T, gammas: Array<f64, Ix1>) -> Result<Self, &'static str> {
        if state0.n() != gammas.len() {
            return Err("When creating new DynStateDiscounter: gammas must have length == n");
        }
        Ok(DynStateDiscounter { state0, gammas, _phantoms: PhantomData })
    }
}

impl<A, S, P, T> StateIterator<A, S> for DynStateDiscounter<A, S, P, T>
where A: ActionType,
      S: StrategyType<Act = A>,
      P: PayoffFunc<Act = A>,
      T: State<P> + MutatesOnAction<A>
{
    type PFunc = P;
    type StateType = T;
    fn state0(&self) -> &T {
        &self.state0
    }

    fn advance_state(&self, state: &mut T, actions: &A) {
        state.mutate_on_action_inplace(actions);
    }
}

impl<A, S, P, T> Discounter for DynStateDiscounter<A, S, P, T>
where A: ActionType,
      S: StrategyType<Act = A>,
      P: PayoffFunc<Act = A>,
      T: State<P> + MutatesOnAction<A>
{
    fn gammas(&self) -> &Array<f64, Ix1> {
        &self.gammas
    }
}

pub type InvestExpDiscounter<P> = DynStateDiscounter<InvestActions, InvestStrategies, P, P>;
