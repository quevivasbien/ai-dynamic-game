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

pub mod pybindings;


#[pymodule]
fn dynapai(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<pybindings::PyActions>()?;
    m.add_class::<pybindings::PyStrategies>()?;
    m.add_class::<pybindings::PyDefaultProd>()?;
    m.add_class::<pybindings::PyLinearReward>()?;
    m.add_class::<pybindings::PyDefaultPayoff>()?;
    m.add_class::<pybindings::PyExponentialDiscounter>()?;
    m.add_class::<pybindings::PySolverOptions>()?;
    m.add_function(wrap_pyfunction!(pybindings::solve_py, m)?)?;
    Ok(())
}
