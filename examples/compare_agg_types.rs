// test EndsOnContestWin payoff aggregator & compare with InvestExpDiscounter

extern crate numpy;
extern crate dynapai;

use dynapai::csf::MaybeNoWinCSF;
use dynapai::solve::{solve, SolverOptions};
use dynapai::states::{InvestExpDiscounter, EndsOnContestWin};
use numpy::ndarray::{Array};

use dynapai::cost_func::FixedInvestCost;
use dynapai::disaster_cost::ConstantDisasterCost;
use dynapai::prod_func::DefaultProd;
use dynapai::payoff_func::DefaultPayoff;
use dynapai::reward_func::LinearReward;
use dynapai::risk_func::WinnerOnlyRisk;

const NSTEPS: usize = 10;

fn main() {
    let prod_func = DefaultProd::new(
        Array::from_vec(vec![10., 10.]),
        Array::from_vec(vec![0.5, 0.5]),
        Array::from_vec(vec![10., 10.]),
        Array::from_vec(vec![0.5, 0.5]),
    ).unwrap();
    let payoff_func = DefaultPayoff::new(
        prod_func,
        WinnerOnlyRisk::new(2, 0.5),
        MaybeNoWinCSF::default(),
        LinearReward::default(2),
        ConstantDisasterCost::new(2, 1.),
        FixedInvestCost::from_elems(2, 0.1, 0.1),
    ).unwrap();

    let agg1 = InvestExpDiscounter::new(
        payoff_func, 
        Array::from_vec(vec![0.9, 0.8])
    ).unwrap();

    let options = SolverOptions::random_init(NSTEPS);

    let res1 = solve(&agg1, &options).unwrap();
    println!("Got result:\n{}\n", res1);

    let agg2 = EndsOnContestWin::new(agg1);

    let res2 = solve(&agg2, &options).unwrap();
    println!("Got result:\n{}\n", res2);

}
