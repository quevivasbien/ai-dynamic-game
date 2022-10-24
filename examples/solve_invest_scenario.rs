extern crate numpy;
extern crate dynapai;

use numpy::ndarray::{Array};
use dynapai::*;

const NSTEPS: usize = 10;
const NTHREADS: usize = 4;

fn main() {
    let pfunc = payoff_func::DefaultPayoff::new(
        prod_func::DefaultProd::new(
            Array::from_vec(vec![10., 10.]),
            Array::from_vec(vec![0.5, 0.5]),
            Array::from_vec(vec![10., 10.]),
            Array::from_vec(vec![0.5, 0.5]),
        ).unwrap(),
        risk_func::WinnerOnlyRisk::new(2, 0.5),
        csf::DefaultCSF,
        reward_func::LinearReward::default(2),
        disaster_cost::ConstantDisasterCost::new(2, 1.),
        cost_func::FixedInvestCost::from_elems(2, 0.1, 0.1),
    ).unwrap();

    let agg = states::InvestExponentialDiscounter::new(
        pfunc, 
        vec![0.9, 0.8]
    ).unwrap();

    // solve the same problem NTHREADS times in parallel
    let scenario = scenarios::Scenario::new(vec![agg; NTHREADS]);
    let options = solve::SolverOptions::random_init(NSTEPS);
    let res = scenario.solve(&options).unwrap();
    println!("Got result:");
    for (i, r) in res.iter().enumerate() {
        println!("Problem {} of {}:\n{}\n", i + 1, res.len(), r);
    }
}