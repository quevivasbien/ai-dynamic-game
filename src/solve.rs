use argmin::core::CostFunction;
use argmin_math::{ArgminAdd, ArgminSub, ArgminMul};

use crate::strategies::*;
use crate::states::PayoffAggregator;
use crate::utils::isapprox_arr;

#[derive(Clone, Debug)]
pub struct SolverOptions {
    init_guess: Strategies,
    max_iters: u32,
    iter_tol: f64,
}

struct PlayerObjective<'a, T: PayoffAggregator>{
    pub payoff_aggregator: &'a T,
    pub i: usize,
}

// implement traits needed for argmin

impl<T: PayoffAggregator> CostFunction for PlayerObjective<'_, T> {
    type Param = Strategies;
    type Output = f64;

    fn cost(&self, params: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
        Ok(-self.payoff_aggregator.u_i(self.i, params))
    }
}

// allow addition of two strategy sets
impl ArgminAdd<Strategies, Strategies> for Strategies {
    fn add(&self, other: &Self) -> Self {
        Strategies::from_array(&self.x + &other.x)
    }
}

// allow subtraction of two strategy sets
impl ArgminSub<Strategies, Strategies> for Strategies {
    fn sub(&self, other: &Self) -> Self {
        Strategies::from_array(&self.x - &other.x)
    }
}

// allow multiplying a strategy set by a scalar
impl ArgminMul<f64, Strategies> for Strategies {
    fn mul(&self, other: &f64) -> Self {
        Strategies::from_array(*other * &self.x)
    }
}

fn update_strat<T: PayoffAggregator>(strat: &mut Strategies, agg: &T) {
    // TODO: This, obviously
}

fn within_tol(current: &Strategies, last: &Strategies, tol: f64) -> bool {
    isapprox_arr(current.x.view(), last.x.view(), tol, f64::EPSILON.sqrt())
}

pub fn solve<T: PayoffAggregator>(agg: &T, options: &SolverOptions) -> Strategies {
    let mut current_strat = options.init_guess.clone();
    for _i in 0..options.max_iters {
        let last_strat = current_strat.clone();
        update_strat(&mut current_strat, agg);
        if within_tol(&current_strat, &last_strat, options.iter_tol) {
            break;
        }
    }
    current_strat
}
