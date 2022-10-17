use numpy::ndarray::{Array, ArrayView, Ix2, s};
use argmin::core::{CostFunction, Executor};
use argmin::solver::neldermead::NelderMead;

use crate::states::PayoffAggregator;
use crate::strategies::*;
use crate::utils::isapprox_arr;

#[derive(Clone, Debug)]
pub struct SolverOptions<S: StrategyType> {
    pub init_guess: S,
    pub max_iters: u64,
    pub tol: f64,
    pub nm_options: NMOptions,
}

#[derive(Clone, Debug)]
pub struct NMOptions {
    pub init_simplex_size: f64,
    pub max_iters: u64,
    pub tol: f64,
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
                (self.base_strategies.t(), self.base_strategies.n()),
                params.iter().map(|x| x.exp()).collect(),
            ).unwrap()
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

fn solve_for_i<S: StrategyType, T: PayoffAggregator<Strat = S>>(i: usize, strat: &S, agg: &T, options: &NMOptions) -> Result<S, argmin::core::Error> {
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
    let mut new_strat = strat.clone();
    new_strat.data_mut().slice_mut(s![.., i, ..]).assign(
        &Array::from_shape_vec(
            (strat.t(), strat.n()),
            res.state.best_param.unwrap().iter().map(|x| x.exp()).collect(),
        )?
    );
    Ok(new_strat)
}

fn update_strat<S: StrategyType, T: PayoffAggregator<Strat = S>>(strat: &mut S, agg: &T, nm_options: &NMOptions) -> Result<(), argmin::core::Error> {
    for i in 0..strat.n() {
        let new_strat = solve_for_i(i, strat, agg, nm_options)?;
        strat.data_mut().slice_mut(s![.., i, ..]).assign(&new_strat.data().slice(s![.., i, ..]));
    }
    Ok(())
}

fn within_tol<S: StrategyType>(current: &S, last: &S, tol: f64) -> bool {
    isapprox_arr(current.data().view(), last.data().view(), tol, f64::EPSILON.sqrt())
}

pub fn solve<S: StrategyType, T: PayoffAggregator<Strat = S>>(agg: &T, options: &SolverOptions<S>) -> Result<S, argmin::core::Error> {
    let mut current_strat = options.init_guess.clone();
    for _i in 0..options.max_iters {
        let last_strat = current_strat.clone();
        update_strat(&mut current_strat, agg, &options.nm_options)?;
        if within_tol(&current_strat, &last_strat, options.tol) {
            println!("Exited on iteration {}", _i);
            return Ok(current_strat);
        }
    }
    println!("Reached max iterations ({})", options.max_iters);
    Ok(current_strat)
}
