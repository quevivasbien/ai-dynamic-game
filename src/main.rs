pub mod utils;

pub mod strategies;

pub mod cost_func;
pub mod csf;
pub mod disaster_cost;
pub mod payoff_func;
pub mod prod_func;
pub mod reward_func;
pub mod risk_func;

pub mod states;

pub mod solve;

use ndarray::Array;
use std::rc::Rc;

use crate::strategies::{Actions, Strategies};
use payoff_func::PayoffFunc;
use states::PayoffAggregator;

fn main() {
    let prod_func = prod_func::DefaultProd {
        a: Array::from_vec(vec![10.0, 10.0]),
        alpha: Array::from_vec(vec![0.5, 0.5]),
        b: Array::from_vec(vec![10.0, 10.0]),
        beta: Array::from_vec(vec![0.5, 0.5]),
    };

    let risk_func = risk_func::WinnerOnlyRisk {
        theta: Array::from_vec(vec![0.5, 0.5]),
    };

    let csf = csf::DefaultCSF;

    let reward_func = reward_func::LinearReward::new(2);

    let disaster_cost = disaster_cost::ConstantDisasterCost {
        d: Array::from_vec(vec![1.0, 1.0]),
    };

    let cost_func = cost_func::FixedUnitCost::new(2, 0.1);

    let payoff_func = payoff_func::DefaultPayoff {
        prod_func,
        risk_func,
        csf,
        reward_func,
        disaster_cost,
        cost_func,
    };

    let xs = Array::from_vec(vec![1.0, 1.0]);
    let xp = Array::from_vec(vec![1.0, 1.0]);

    let actions = Actions::from_inputs(xs, xp);

    println!("Payoff from single period of consumption:");
    println!("{}", payoff_func.u(&actions));

    let state = Rc::new(states::CommonBeliefs {
        belief: Box::new(payoff_func)
    });

    let agg = states::ExponentialDiscounter {
        states: vec![state.clone(), state.clone()],
        gammas: vec![0.95, 0.80],
    };

    let strategies = Strategies::from_actions(vec![actions.clone(), actions.clone()]);

    println!("Payoff from multiple periods with discounting:");
    println!("{}", agg.u(&strategies));

    let sol = solve::solve(
        &agg,
        &solve::SolverOptions {
            init_guess: strategies,
            max_iters: 100,
            tol: 1e-6,
            nm_options: solve::NMOptions {
                init_simplex_size: 1.0,
                max_iters: 100,
                tol: 1e-8,
            }
        }
    );

    println!("Result from solving:\n{}", sol);

}
