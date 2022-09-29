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

use std::rc::Rc;

use crate::strategies::{Actions, Strategies};
use payoff_func::PayoffFunc;
use states::PayoffAggregator;

fn main() {
    let prod_func = prod_func::DefaultProd {
        a: vec![10.0, 10.0],
        alpha: vec![0.5, 0.5],
        b: vec![10.0, 10.0],
        beta: vec![0.5, 0.5],
    };

    let risk_func = risk_func::WinnerOnlyRisk {
        theta: vec![0.5, 0.5],
    };

    let csf = csf::DefaultCSF;

    let reward_func = reward_func::LinearReward::new(2);

    let disaster_cost = disaster_cost::ConstantDisasterCost {
        d: vec![1.0, 1.0],
    };

    let cost_func = cost_func::DefaultCost::new(2, 0.1);

    let payoff_func = payoff_func::DefaultPayoff {
        prod_func,
        risk_func,
        csf,
        reward_func,
        disaster_cost,
        cost_func,
    };

    let xs = vec![1.0, 1.0];
    let xp = vec![1.0, 1.0];

    let actions = Actions::new(xs, xp);

    println!("Result from single-period of consumption:");
    println!("{:?}", payoff_func.u(&actions));

    let state = Rc::new(states::CommonBeliefs {
        belief: Box::new(payoff_func)
    });

    let agg = states::ExponentialDiscounter {
        states: vec![state.clone(), state.clone()],
        gammas: vec![0.95, 0.80],
    };

    let strategies = Strategies::new(vec![actions.clone(), actions.clone()]);

    println!("Result from multiple periods with discounting:");
    println!("{:?}", agg.u(&strategies));

}
