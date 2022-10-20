use std::sync::Arc;
use rayon::prelude::*;

use crate::states::PayoffAggregator;
use crate::solve::{solve, SolverOptions};

pub struct Scenario<T: PayoffAggregator>(Vec<Arc<T>>);

impl<T: PayoffAggregator> Scenario<T> {

    pub fn new(aggs: Vec<Arc<T>>) -> Self {
        Scenario(aggs)
    }

    pub fn solve(&self, options: &SolverOptions<T::Strat>) -> Result<Vec<T::Strat>, argmin::core::Error> {
        self.0.par_iter().map(|agg| solve(agg.as_ref(), options)).collect()
    }
}
