use::pyo3::prelude::*;

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
pub mod scenarios;

pub mod pybindings;
use pybindings::*;

#[pymodule]
fn dynapai(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyActions>()?;
    m.add_class::<PyStrategies>()?;
    m.add_class::<PyDefaultProd>()?;
    m.add_class::<PyLinearReward>()?;
    m.add_class::<PyDefaultPayoff>()?;
    m.add_class::<PySolverOptions>()?;
    m.add_class::<PyExponentialDiscounter>()?;
    m.add_class::<PyInvestActions>()?;
    m.add_class::<PyInvestStrategies>()?;
    m.add_class::<PyInvestProd>()?;
    m.add_class::<PyInvestPayoff>()?;
    m.add_class::<PyInvestExpDiscounter>()?;
    m.add_class::<PyEndOnWinAggregator>()?;
    m.add_class::<PyInvestEndOnWinAggregator>()?;
    m.add_class::<PyScenario>()?;
    m.add_class::<PyInvestScenario>()?;
    Ok(())
}
