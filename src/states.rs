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

pub trait PayoffAggregator: Send + Sync {
    type Strat: StrategyType;
    fn n(&self) -> usize;
    fn u_i(&self, i: usize, strategies: &Self::Strat) -> f64;
    fn u(&self, strategies: &Self::Strat) -> Array<f64, Ix1> {
        Array::from_iter((0..strategies.n()).map(|i| self.u_i(i, strategies)))
    }
}

#[derive(Clone)]
pub struct ExponentialDiscounter<U: PayoffFunc<Act = Actions>, T: State<U>> {
    n: usize,
    pub states: Vec<T>,
    pub gammas: Array<f64, Ix1>,
    phantom: std::marker::PhantomData<U>,
}

impl<U: PayoffFunc<Act = Actions>, T: State<U>> ExponentialDiscounter<U, T> {
    pub fn new(states: Vec<T>, gammas: Array<f64, Ix1>) -> Result<Self, &'static str> {
        if states.len() == 0 {
            return Err("When creating new ExponentialDiscounter: states must have length > 0");
        }
        let n = states[0].n();
        if states.iter().any(|s| s.n() != n){
            return Err("When creating new ExponentialDiscounter: All states must have the same n");
        }
        if n != gammas.len() {
            return Err("When creating new ExponentialDiscounter: gammas must have length == n");
        }
        Ok(ExponentialDiscounter {
            n,
            states,
            gammas,
            phantom: std::marker::PhantomData,
        })
    }
    pub fn new_static(state0: T, t: usize, gammas: Array<f64, Ix1>) -> Result<Self, &'static str> {
        if state0.n() != gammas.len() {
            return Err("When creating new ExponentialDiscounter: state0.n() must match gammas.len()");
        }
        Ok(Self::new(vec![state0; t], gammas)?)
    }
}

impl<U: PayoffFunc<Act = Actions>, T: State<U>> PayoffAggregator for ExponentialDiscounter<U, T> {
    type Strat = Strategies;

    fn n(&self) -> usize {
        self.n
    }

    fn u(&self, strategies: &Strategies) -> Array<f64, Ix1> {
        assert_eq!(self.states.len(), strategies.t(), "Number of states should match number of time periods in strategies");
        let actions_seq = strategies.clone().to_actions();
        actions_seq.iter().enumerate().map(|(t, actions)| {
            let states = &self.states[t];
            Array::from_iter(self.gammas.iter().enumerate().map(|(i, gamma)| {
                gamma.powi(t.try_into().unwrap()) * states.belief(i).u_i(i, actions)
            }))
        }).fold(Array::zeros(self.gammas.len()), |acc, x| acc + x)
    }

    fn u_i(&self, i: usize, strategies: &Strategies) -> f64 {
        assert_eq!(self.states.len(), strategies.t(), "Number of states should match number of time periods in strategies");
        let actions_seq = strategies.clone().to_actions();
        actions_seq.iter().enumerate().map(|(t, actions)| {
            let gamma = self.gammas[i];
            let belief = self.states[t].belief(i);
            gamma.powi(t.try_into().unwrap()) * belief.u_i(i, actions)
        }).sum()
    }
}


#[derive(Clone)]
pub struct InvestExponentialDiscounter<T>
where T: PayoffFunc<Act = InvestActions> + MutatesOnAction<InvestActions>
{
    pub state0: T,
    pub gammas: Array<f64, Ix1>,
}

impl<T> InvestExponentialDiscounter<T>
where T: PayoffFunc<Act = InvestActions> + MutatesOnAction<InvestActions>
{
    pub fn new(state0: T, gammas: Array<f64, Ix1>) -> Result<Self, &'static str> {
        if gammas.len() != state0.n() {
            return Err("When creating new InvestExponentialDiscounter: gammas must have length equal to state0.n()");
        }
        Ok(InvestExponentialDiscounter { state0, gammas })
    }
}

impl<T> PayoffAggregator for InvestExponentialDiscounter<T>
where T: PayoffFunc<Act = InvestActions> + MutatesOnAction<InvestActions>
{
    type Strat = InvestStrategies;

    fn n(&self) -> usize {
        self.state0.n()
    }

    fn u(&self, strategies: &InvestStrategies) -> Array<f64, Ix1> {
        let actions_seq = strategies.clone().to_actions();
        let mut state = self.state0.clone();
        let mut out: Array<f64, Ix1> = Array::zeros(self.gammas.len());
        for (t, actions) in actions_seq.iter().enumerate() {
            out.iter_mut().zip(self.gammas.iter()).enumerate().for_each(|(i, (out_i, gamma))| {
                *out_i += gamma.powi(t.try_into().unwrap()) * state.belief(i).u_i(i, actions);
            });
            if t != actions_seq.len() - 1 {
                state.mutate_on_action_inplace(actions);
            }
        }
        out
    }

    fn u_i(&self, i: usize, strategies: &InvestStrategies) -> f64 {
        let actions_seq = strategies.clone().to_actions();
        let mut state = self.state0.clone();
        let mut out = 0.;
        for (t, actions) in actions_seq.iter().enumerate() {
            let gamma = self.gammas[i];
            let belief = state.belief(i);
            out += gamma.powi(t.try_into().unwrap()) * belief.u_i(i, actions);
            if t != actions_seq.len() - 1 {
                state.mutate_on_action_inplace(actions);
            }
        }
        out
    }
}
