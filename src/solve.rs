use numpy::ndarray::{Array, ArrayView, Ix2, s};
use argmin::core::{CostFunction, Executor};
use argmin::solver::neldermead::NelderMead;
use rayon::prelude::*;

use crate::states::PayoffAggregator;
use crate::strategies::*;
use crate::utils::isapprox_iters;

const INIT_MU: f64 = -1.;
const INIT_SIGMA: f64 = 0.1;

#[derive(Clone, Debug)]
pub enum InitGuess<S: StrategyType> {
    Random(usize),
    Fixed(S),
}

impl<S: StrategyType> InitGuess<S> {
    fn to_fixed(&self, n: usize) -> S {
        match self {
            InitGuess::Random(t) => S::random(*t, n, INIT_MU, INIT_SIGMA).unwrap(),
            InitGuess::Fixed(x) => x.clone(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct SolverOptions<S: StrategyType> {
    pub init_guess: InitGuess<S>,
    pub max_iters: u64,
    pub tol: f64,
    pub nm_options: NMOptions,
}

impl<S: StrategyType> SolverOptions<S> {
    pub fn from_init_guess(init_guess: S) -> Self {
        SolverOptions {
            init_guess: InitGuess::Fixed(init_guess),
            max_iters: 200,
            tol: 1e-6,
            nm_options: NMOptions::default(),
        }
    }

    pub fn random_init(t: usize) -> Self {
        SolverOptions {
            init_guess: InitGuess::Random(t),
            max_iters: 200,
            tol: 1e-6,
            nm_options: NMOptions::default(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct NMOptions {
    pub init_simplex_size: f64,
    pub max_iters: u64,
    pub tol: f64,
}

impl Default for NMOptions {
    fn default() -> Self {
        NMOptions {
            init_simplex_size: 0.1,
            max_iters: 200,
            tol: 1e-8,
        }
    }
}

struct PlayerObjective<'a, S: StrategyType, T: PayoffAggregator<Strat = S>>{
    pub payoff_aggregator: &'a T,
    pub i: usize,
    pub base_strategies: &'a S,
}

// implement traits needed for argmin

impl<S: StrategyType, T: PayoffAggregator<Strat = S>> CostFunction for PlayerObjective<'_, S, T> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, params: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
        let mut strategies = self.base_strategies.clone();
        strategies.data_mut().slice_mut(s![.., self.i, ..]).assign(
            &Array::from_shape_vec(
                (self.base_strategies.t(), S::nparams()),
                params.iter().map(|x| x.exp()).collect(),
            )?
        );
        Ok(-self.payoff_aggregator.u_i(self.i, &strategies))
    }
}

fn create_simplex(init_guess: ArrayView<f64, Ix2>, init_simplex_size: f64) -> Vec<Vec<f64>> {
    let mut simplex = Vec::new();
    let base: Vec<f64> = init_guess.iter().map(|x| x.ln()).collect();
    for i in 0..base.len() {
        let mut x = base.clone();
        x[i] += init_simplex_size;
        simplex.push(x);
    }
    simplex.push(base);
    simplex
}

fn solve_for_i<S: StrategyType, T: PayoffAggregator<Strat = S>>(i: usize, strat: &S, agg: &T, options: &NMOptions) -> Result<Array<f64, Ix2>, argmin::core::Error> {
    let init_simplex = create_simplex(
        strat.data().slice(s![.., i, ..]),
        options.init_simplex_size
    );
    let obj = PlayerObjective {
        payoff_aggregator: agg,
        i,
        base_strategies: strat,
    };
    let solver = NelderMead::new(init_simplex).with_sd_tolerance(options.tol).unwrap();
    let res = Executor::new(obj, solver)
        .configure(|state| state.max_iters(options.max_iters))
        .run()?;
    Ok(Array::from_shape_vec(
        (strat.t(), S::nparams()),
        res.state.best_param.unwrap().iter().map(|x| x.exp()).collect(),
    )?)
}

fn update_strat<S, T>(strat: &mut S, agg: &T, nm_options: &NMOptions) -> Result<(), argmin::core::Error>
where S: StrategyType, T: PayoffAggregator<Strat = S>
{
    let new_data = (0..strat.n()).into_par_iter().map(|i| {
        solve_for_i(i, strat, agg, nm_options)
    }).collect::<Result<Vec<_>,_>>()?;
    for (i, x) in new_data.iter().enumerate() {
        strat.data_mut().slice_mut(s![.., i, ..]).assign(x);
    }
    Ok(())
}

fn within_tol<S: StrategyType>(current: &S, last: &S, tol: f64) -> bool {
    isapprox_iters(
        current.data().iter().map(|x| x.ln()),
        last.data().iter().map(|x| x.ln()),
        tol, f64::EPSILON.sqrt()
    )
}

pub fn solve<S, T>(agg: &T, options: &SolverOptions<S>) -> Result<S, argmin::core::Error>
where S: StrategyType, T: PayoffAggregator<Strat = S>
{
    let mut current_strat = options.init_guess.to_fixed(agg.n());
    for i in 0..options.max_iters {
        let last_strat = current_strat.clone();
        update_strat(&mut current_strat, agg, &options.nm_options)?;
        if within_tol(&current_strat, &last_strat, options.tol) {
            println!("Exited on iteration {}", i);
            return Ok(current_strat);
        }
    }
    println!("Reached max iterations ({})", options.max_iters);
    Ok(current_strat)
}
