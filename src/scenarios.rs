use std::marker::PhantomData;

use rayon::prelude::*;

use crate::strategies::{ActionType, StrategyType};
use crate::states::PayoffAggregator;
use crate::solve::{solve, SolverOptions};

pub struct Scenario<A, S, T>
where A: ActionType, S: StrategyType<Act = A>, T: PayoffAggregator<A, S>
{
    n: usize,
    aggs: Vec<T>,
    _phantoms: PhantomData<(A, S)>
}

impl<A, S, T> Scenario<A, S, T>
where A: ActionType, S: StrategyType<Act = A>, T: PayoffAggregator<A, S>
{

    pub fn new(aggs: Vec<T>) -> Result<Self, &'static str> {
        let n = aggs[0].n();
        if aggs.iter().any(|agg| agg.n() != n) {
            return Err("All payoff aggregators must have the same number of players");
        }
        Ok(Scenario {
            n,
            aggs,
            _phantoms: PhantomData
        })
    }

    pub fn aggs(&self) -> &Vec<T> {
        &self.aggs
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn solve(&self, options: &SolverOptions<S>) -> Result<Vec<S>, argmin::core::Error> {
        self.aggs.par_iter().map(|agg| solve(agg, options)).collect()
    }
}
