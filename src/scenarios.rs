use rayon::prelude::*;

use crate::states::PayoffAggregator;
use crate::solve::{solve, SolverOptions};

pub struct Scenario<T: PayoffAggregator>(Vec<T>);

impl<T: PayoffAggregator> Scenario<T> {

    pub fn new(aggs: Vec<T>) -> Self {
        Scenario(aggs)
    }

    pub fn solve(&self, options: &SolverOptions<T::Strat>) -> Result<Vec<T::Strat>, argmin::core::Error> {
        self.0.par_iter().map(|agg| solve(agg, options)).collect()
    }
}
