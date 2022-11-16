use std::marker::PhantomData;

use numpy::Ix2;
use numpy::ndarray::{Array, Ix1};

use crate::cost_func::CostFunc;
use crate::csf::CSF;
use crate::disaster_cost::DisasterCost;
use crate::prod_func::ProdFunc;
use crate::reward_func::RewardFunc;
use crate::risk_func::RiskFunc;
use crate::strategies::*;
use crate::payoff_func::{PayoffFunc, DefaultPayoff};

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
            if t != strategies.t() - 1 {
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
            if t != strategies.t() - 1 {
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
    pub state: T,
    pub gammas: Array<f64, Ix1>,
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
    pub state0: T,
    pub gammas: Array<f64, Ix1>,
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


pub struct EndsOnContestWin<A, S, T, U, V, W, X, Y, Z, C>
where A: ActionType,
      S: StrategyType<Act = A>,
      T: ProdFunc<A>,
      U: RiskFunc,
      V: CSF,
      W: RewardFunc,
      X: DisasterCost,
      Y: CostFunc<A>,
      Z: State<DefaultPayoff<A, T, U, V, W, X, Y>>,
      C: Discounter + StateIterator<A, S>
{
    pub child: C,
    _phantoms: PhantomData<(A, S, T, U, V, W, X, Y, Z)>,
}

impl<A, S, T, U, V, W, X, Y, Z, C> EndsOnContestWin<A, S, T, U, V, W, X, Y, Z, C>
where A: ActionType,
      S: StrategyType<Act = A>,
      T: ProdFunc<A>,
      U: RiskFunc,
      V: CSF,
      W: RewardFunc,
      X: DisasterCost,
      Y: CostFunc<A>,
      Z: State<DefaultPayoff<A, T, U, V, W, X, Y>> + MutatesOnAction<A>,
      C: Discounter + StateIterator<A, S, StateType = Z>
{
    pub fn new(child: C) -> Self {
        EndsOnContestWin { child, _phantoms: PhantomData }
    }
    pub fn probas(&self, strategies: &S) -> Array<f64, Ix2> {
        let mut probas = vec![1.; self.n()];
        let mut all_probas: Vec<f64> = Vec::with_capacity(self.n() * strategies.t());
        let actions_seq = strategies.clone().to_actions();
        let mut state = self.child.state0().clone();
        for (t, actions) in actions_seq.iter().enumerate() {
            for i in 0..self.n() {
                all_probas.push(probas[i]);
                if t != strategies.t() - 1 {
                    let payoff_func = state.belief(i);
                    let (_, p) = payoff_func.prod_func.f(actions);
                    probas[i] *= 1. - payoff_func.csf.q(p.view()).iter().sum::<f64>();
                    state.mutate_on_action_inplace(actions);
                }
            }
        }
        Array::from_shape_vec((strategies.t(), self.n()), all_probas).unwrap()
    }
}

impl<A, S, T, U, V, W, X, Y, Z, C> StateIterator<A, S> for EndsOnContestWin<A, S, T, U, V, W, X, Y, Z, C>
where A: ActionType,
      S: StrategyType<Act = A>,
      T: ProdFunc<A>,
      U: RiskFunc,
      V: CSF,
      W: RewardFunc,
      X: DisasterCost,
      Y: CostFunc<A>,
      Z: State<DefaultPayoff<A, T, U, V, W, X, Y>> + MutatesOnAction<A>,
      C: Discounter + StateIterator<A, S, StateType = Z>,
{
    type PFunc = DefaultPayoff<A, T, U, V, W, X, Y>;
    type StateType = Z;
    fn state0(&self) -> &Z {
        self.child.state0()
    }

    fn advance_state(&self, state: &mut Z, actions: &A) {
        state.mutate_on_action_inplace(actions);
    }
}

impl<A, S, T, U, V, W, X, Y, Z, C> PayoffAggregator<A, S> for EndsOnContestWin<A, S, T, U, V, W, X, Y, Z, C>
where A: ActionType,
      S: StrategyType<Act = A>,
      T: ProdFunc<A>,
      U: RiskFunc,
      V: CSF,
      W: RewardFunc,
      X: DisasterCost,
      Y: CostFunc<A>,
      Z: State<DefaultPayoff<A, T, U, V, W, X, Y>> + MutatesOnAction<A>,
      C: Discounter + StateIterator<A, S, StateType = Z>,
{
    fn n(&self) -> usize {
        self.child.n()
    }
    fn u_i(&self, i: usize, strategies: &S) -> f64 {
        let actions_seq = strategies.clone().to_actions();
        let mut state = self.child.state0().clone();
        let gamma = self.child.gammas()[i];
        let mut proba = 1.;  // probability that nobody has won yet
        let mut u = 0.;
        for (t, actions) in actions_seq.iter().enumerate() {
            let payoff_func = state.belief(i);
            u += proba * gamma.powi(t.try_into().unwrap()) * payoff_func.u_i(i, actions);
            if t != strategies.t() - 1 {
                // update proba
                let (_, p) = payoff_func.prod_func.f(actions);
                proba *= 1. - payoff_func.csf.q(p.view()).iter().sum::<f64>();
                // update state
                self.advance_state(&mut state, actions);
            }
        }
        u
    }
    fn u(&self, strategies: &S) -> Array<f64, Ix1> {
        let actions_seq = strategies.clone().to_actions();
        let state = &mut self.state0().clone();
        let gammas = self.child.gammas();
        let mut probas = vec![1.; gammas.len()];
        let mut u: Array<f64, Ix1> = Array::zeros(gammas.len());
        for (t, actions) in actions_seq.iter().enumerate() {
            u.iter_mut().zip(gammas.iter()).enumerate().for_each(|(i, (u_i, gamma))| {
                let payoff_func = state.belief(i);
                // update u
                *u_i += probas[i] * gamma.powi(t.try_into().unwrap()) * payoff_func.u_i(i, actions);
                if t != strategies.t() {
                    // update probas
                    let (_, p) = payoff_func.prod_func.f(actions);
                    probas[i] *= 1. - payoff_func.csf.q(p.view()).iter().sum::<f64>();
                }
            });
            if t != strategies.t() - 1 {
                // update state
                self.advance_state(state, actions);
            }
        }
        u
    }
}
