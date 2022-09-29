use argmin::core::CostFunction;
use argmin_math::{ArgminAdd, ArgminSub, ArgminMul};

use crate::strategies::*;
use crate::states::PayoffAggregator;
use crate::utils::isapprox_vec;

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
        assert_eq!(self.t, other.t, "Strategy sets must have same length");
        assert_eq!(self.n, other.n, "Strategy sets must have same number of players");
        let x = self.x.iter().zip(other.x.iter()).map(|(a, b)| {
            let xs = a.xs.iter().zip(b.xs.iter()).map(|(x, y)| x + y).collect();
            let xp = a.xp.iter().zip(b.xp.iter()).map(|(x, y)| x + y).collect();
            Actions::new(xs, xp)
        }).collect();
        Strategies::new(x)
    }
}

// allow subtraction of two strategy sets
impl ArgminSub<Strategies, Strategies> for Strategies {
    fn sub(&self, other: &Self) -> Self {
        assert_eq!(self.t, other.t, "Strategy sets must have same length");
        assert_eq!(self.n, other.n, "Strategy sets must have same number of players");
        let x = self.x.iter().zip(other.x.iter()).map(|(a, b)| {
            let xs = a.xs.iter().zip(b.xs.iter()).map(|(x, y)| x - y).collect();
            let xp = a.xp.iter().zip(b.xp.iter()).map(|(x, y)| x - y).collect();
            Actions::new(xs, xp)
        }).collect();
        Strategies::new(x)
    }
}

// allow multiplying a strategy set by a scalar
impl ArgminMul<f64, Strategies> for Strategies {
    fn mul(&self, other: &f64) -> Self {
        let x = self.x.iter().map(|a| {
            let xs = a.xs.iter().map(|x| x * other).collect();
            let xp = a.xp.iter().map(|x| x * other).collect();
            Actions::new(xs, xp)
        }).collect();
        Strategies::new(x)
    }
}

fn update_strat<T: PayoffAggregator>(strat: &mut Strategies, agg: &T) {
    // TODO: This, obviously
}

fn within_tol(current: &Strategies, last: &Strategies, tol: f64) -> bool {
    current.x.iter().zip(last.x.iter()).all(|(c, l)|
        isapprox_vec(&c.xs, &l.xs, tol, f64::EPSILON.sqrt())
        &&
        isapprox_vec(&c.xp, &l.xp, tol, f64::EPSILON.sqrt())
    )
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
