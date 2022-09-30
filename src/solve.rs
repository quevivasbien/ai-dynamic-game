use ndarray::{Array, ArrayView, Ix2, s};
use argmin::core::{CostFunction, Executor};
use argmin::solver::neldermead::NelderMead;

use crate::strategies::*;
use crate::states::PayoffAggregator;
use crate::utils::isapprox_arr;

#[derive(Clone, Debug)]
pub struct SolverOptions {
    init_guess: Strategies,
    max_iters: u64,
    tol: f64,
    nm_options: NMOptions,
}

#[derive(Clone, Debug)]
pub struct NMOptions {
    pub init_simplex_size: f64,
    pub max_iters: u64,
    pub tol: f64,
}

struct PlayerObjective<'a, T: PayoffAggregator>{
    pub payoff_aggregator: &'a T,
    pub i: usize,
    pub base_strategies: &'a Strategies,
}

// implement traits needed for argmin

impl<T: PayoffAggregator> CostFunction for PlayerObjective<'_, T> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, params: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
        let mut strategies = self.base_strategies.clone();
        strategies.x.slice_mut(s![.., self.i, ..]).assign(
            &Array::from_shape_vec(
                (self.base_strategies.t, self.base_strategies.n),
                params.clone()
            ).unwrap()
        );
        Ok(-self.payoff_aggregator.u_i(self.i, &strategies))
    }
}

fn create_simplex(init_guess: ArrayView<f64, Ix2>, init_simplex_size: f64) -> Vec<Vec<f64>> {
    let mut simplex = Vec::new();
    let base: Vec<f64> = init_guess.iter().cloned().collect();
    for i in 0..base.len() {
        let mut x = base.clone();
        x[i] += init_simplex_size;
        simplex.push(x);
    }
    simplex.push(base);
    simplex
}

fn solve_for_i<T: PayoffAggregator>(i: usize, strat: &Strategies, agg: &T, options: &NMOptions) -> Strategies {
    let init_simplex = create_simplex(
        strat.x.slice(s![.., i, ..]),
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
        .run().unwrap();
    println!("Found solution: {}", res);
    let mut new_strat = strat.clone();
    new_strat.x.slice_mut(s![.., i, ..]).assign(
        &Array::from_shape_vec(
            (strat.t, strat.n),
            res.state.best_param.unwrap()
        ).unwrap()
    );
    new_strat
}

fn update_strat<T: PayoffAggregator>(strat: &mut Strategies, agg: &T, nm_options: &NMOptions) {
    for i in 0..strat.n {
        let new_strat = solve_for_i(i, strat, agg, nm_options);
        strat.x.slice_mut(s![.., i, ..]).assign(&new_strat.x.slice(s![.., i, ..]));
    }
}

fn within_tol(current: &Strategies, last: &Strategies, tol: f64) -> bool {
    isapprox_arr(current.x.view(), last.x.view(), tol, f64::EPSILON.sqrt())
}

pub fn solve<T: PayoffAggregator>(agg: &T, options: &SolverOptions) -> Strategies {
    let mut current_strat = options.init_guess.clone();
    for _i in 0..options.max_iters {
        let last_strat = current_strat.clone();
        update_strat(&mut current_strat, agg, &options.nm_options);
        if within_tol(&current_strat, &last_strat, options.tol) {
            break;
        }
    }
    current_strat
}
