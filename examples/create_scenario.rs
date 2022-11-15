extern crate numpy;
extern crate dynapai;

use numpy::ndarray::{Array, Array1};

use dynapai::payoff_func::DefaultPayoff;
use dynapai::{*, strategies::*};
use dynapai::prod_func::DefaultProd;
use dynapai::risk_func::WinnerOnlyRisk;
use dynapai::csf::DefaultCSF;
use dynapai::reward_func::LinearReward;
use dynapai::disaster_cost::ConstantDisasterCost;
use dynapai::cost_func::FixedInvestCost;
use dynapai::states::InvestExpDiscounter;

const NSTEPS: usize = 10;

type InvestPayoff_ = DefaultPayoff<
    InvestActions,
    DefaultProd,
    WinnerOnlyRisk,
    DefaultCSF,
    LinearReward,
    ConstantDisasterCost,
    FixedInvestCost
>;

type InvestDiscounter_ = InvestExpDiscounter<InvestPayoff_>;

fn main() {
    // set up two problems with different values of a
    let prod_funcs = init_rep!(DefaultProd => 
        a: Array1<f64> = vec![Array::from_vec(vec![10., 10.]), Array::from_vec(vec![100., 100.])];
        alpha: Array1<f64> = vec![Array::from_vec(vec![0.5, 0.5])];
        b: Array1<f64> = vec![Array::from_vec(vec![10., 10.])];
        beta: Array1<f64> = vec![Array::from_vec(vec![0.5, 0.5])]
    );
    let payoff_funcs = init_rep!(InvestPayoff_ =>
        prod_func: DefaultProd = prod_funcs;
        risk_func: WinnerOnlyRisk = vec![WinnerOnlyRisk::new(2, 0.5)];
        csf: DefaultCSF = vec![DefaultCSF];
        reward_func: LinearReward = vec![LinearReward::default(2)];
        disaster_cost: ConstantDisasterCost = vec![ConstantDisasterCost::new(2, 1.)];
        cost_func: FixedInvestCost = vec![FixedInvestCost::from_elems(2, 0.1, 0.1)]
    );

    let aggs = init_rep!(InvestDiscounter_ =>
        payoff_func: InvestPayoff_ = payoff_funcs;
        gammas: Array1<f64> = vec![Array::from_vec(vec![0.9, 0.8])]
    );

    // solve the problems in parallel
    let scenario = scenarios::Scenario::new(aggs).unwrap();
    let options = solve::SolverOptions::random_init(NSTEPS);
    let res = scenario.solve(&options).unwrap();
    println!("Got result:");
    for (i, r) in res.iter().enumerate() {
        println!("Problem {} of {}:\n{}\n", i + 1, res.len(), r);
    }
}
