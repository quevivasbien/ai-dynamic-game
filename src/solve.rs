use crate::strategies::*;
use crate::states::PayoffAggregator;
use crate::utils::isapprox_vec;

#[derive(Clone, Debug)]
pub struct SolverOptions {
    init_guess: StrategySet,
    max_iters: u32,
    iter_tol: f64,
}

fn update_strat<T: PayoffAggregator>(strat: &mut StrategySet, agg: &T) {
    // TODO: This, obviously
}

fn within_tol(current: &StrategySet, last: &StrategySet, tol: f64) -> bool {
    // TODO: Implement with proper isapprox function
    current.actions_sequence.iter().zip(last.actions_sequence.iter()).all(|(c, l)|
        isapprox_vec(&c.xs, &l.xs, tol, f64::EPSILON.sqrt())
        &&
        isapprox_vec(&c.xp, &l.xp, tol, f64::EPSILON.sqrt())
    )
}

pub fn solve<T: PayoffAggregator>(agg: &T, options: &SolverOptions) -> StrategySet {
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
