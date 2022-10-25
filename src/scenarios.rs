use rayon::prelude::*;

use crate::states::PayoffAggregator;
use crate::solve::{solve, SolverOptions};

pub struct Scenario<T: PayoffAggregator> {
    n: usize,
    aggs: Vec<T>
}

impl<T: PayoffAggregator> Scenario<T> {

    pub fn new(aggs: Vec<T>) -> Result<Self, &'static str> {
        let n = aggs[0].n();
        if aggs.iter().any(|agg| agg.n() != n) {
            return Err("All payoff aggregators must have the same number of players");
        }
        Ok(Scenario {
            n,
            aggs
        })
    }

    pub fn aggs(&self) -> &Vec<T> {
        &self.aggs
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn solve(&self, options: &SolverOptions<T::Strat>) -> Result<Vec<T::Strat>, argmin::core::Error> {
        self.aggs.par_iter().map(|agg| solve(agg, options)).collect()
    }
}
