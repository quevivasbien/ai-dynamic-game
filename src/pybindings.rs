use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray3, IntoPyArray};
use pyo3::exceptions::PyException;
use pyo3::{prelude::*, types::PyList};

use crate::cost_func::{FixedUnitCost, FixedInvestCost};
use crate::csf::DefaultCSF;
use crate::disaster_cost::ConstantDisasterCost;
use crate::payoff_func::{PayoffFunc, DefaultPayoff};
use crate::prod_func::{ProdFunc, DefaultProd};
use crate::reward_func::LinearReward;
use crate::risk_func::WinnerOnlyRisk;
use crate::solve::{NMOptions, SolverOptions, solve};
use crate::states::{PayoffAggregator, ExponentialDiscounter, InvestExponentialDiscounter};
use crate::strategies::*;


// implement python class containers for Actions and Strategies

#[derive(Clone)]
#[pyclass(name = "Actions")]
pub struct PyActions(Actions);

#[pymethods]
impl PyActions {
    #[new]
    fn from_inputs(xs: PyReadonlyArray1<f64>, xp: PyReadonlyArray1<f64>) -> Self {
        PyActions(Actions::from_inputs(xs.as_array().to_owned(), xp.as_array().to_owned()))
    }

    fn __str__(&self) -> String {
        format!("{}", self.0)
    }
}

#[derive(Clone)]
#[pyclass(name = "InvestActions")]
pub struct PyInvestActions(InvestActions);

#[pymethods]
impl PyInvestActions {
    #[new]
    fn from_inputs(
        xs: PyReadonlyArray1<f64>, xp: PyReadonlyArray1<f64>,
        inv_s: PyReadonlyArray1<f64>, inv_p: PyReadonlyArray1<f64>,
    ) -> Self {
        PyInvestActions(InvestActions::from_inputs(
            xs.as_array().to_owned(), xp.as_array().to_owned(),
            inv_s.as_array().to_owned(), inv_p.as_array().to_owned(),
        ))
    }

    fn __str__(&self) -> String {
        format!("{}", self.0)
    }
}

#[derive(Clone)]
#[pyclass(name = "Strategies")]
pub struct PyStrategies(Strategies);

#[pymethods]
impl PyStrategies {
    #[new]
    fn from_array(x: PyReadonlyArray3<f64>) -> Self {
        PyStrategies(Strategies::from_array(x.as_array().to_owned()))
    }

    #[staticmethod]
    fn from_actions(actions: &PyList) -> Self {
        let actions_vec = actions.iter().map(|a|
            a.extract::<PyActions>()
             .expect("actions should contain only objects of type PyActions").0
        ).collect();
        PyStrategies(Strategies::from_actions(actions_vec))
    }

    fn __str__(&self) -> String {
        format!("{}", self.0)
    }
}

#[derive(Clone)]
#[pyclass(name = "InvestStrategies")]
pub struct PyInvestStrategies(InvestStrategies);

#[pymethods]
impl PyInvestStrategies {
    #[new]
    fn from_array(x: PyReadonlyArray3<f64>) -> Self {
        PyInvestStrategies(InvestStrategies::from_array(x.as_array().to_owned()))
    }

    #[staticmethod]
    fn from_actions(actions: &PyList) -> Self {
        let actions_vec = actions.iter().map(|a|
            a.extract::<PyInvestActions>()
             .expect("actions should contain only objects of type PyInvestActions").0
        ).collect();
        PyInvestStrategies(InvestStrategies::from_actions(actions_vec))
    }

    fn __str__(&self) -> String {
        format!("{}", self.0)
    }
}

// create python class container "ProdFunc" for DefaultProd
#[pyclass(name = "ProdFunc")]
#[derive(Clone)]
pub struct PyDefaultProd(DefaultProd<Actions>);

#[pymethods]
impl PyDefaultProd {
    #[new]
    fn new(
        a: PyReadonlyArray1<f64>, alpha: PyReadonlyArray1<f64>,
        b: PyReadonlyArray1<f64>, beta: PyReadonlyArray1<f64>
    ) -> Self {
        PyDefaultProd(DefaultProd::new(
            a.as_array().to_owned(),
            alpha.as_array().to_owned(),
            b.as_array().to_owned(),
            beta.as_array().to_owned(),
        ).expect("invalid production function parameters"))
    }

    fn f_i(&self, i: usize, actions: &PyActions) -> (f64, f64) {
        self.0.f_i(i, &actions.0)
    }

    fn f<'py>(&self, py: Python<'py>, actions: &PyActions) -> (&'py PyArray1<f64>, &'py PyArray1<f64>) {
        let (s, p) = self.0.f(&actions.0);
        (s.into_pyarray(py), p.into_pyarray(py))
    }
    
    fn with_invest(&self) -> PyInvestProd {
        PyInvestProd(DefaultProd::new(
            self.0.a.to_owned(),
            self.0.alpha.to_owned(),
            self.0.b.to_owned(),
            self.0.beta.to_owned()
        ).expect("invalid production function parameters"))
    }
}

#[pyclass(name = "InvestProdFunc")]
#[derive(Clone)]
pub struct PyInvestProd(DefaultProd<InvestActions>);

#[pymethods]
impl PyInvestProd {
    #[new]
    fn new(
        a: PyReadonlyArray1<f64>, alpha: PyReadonlyArray1<f64>,
        b: PyReadonlyArray1<f64>, beta: PyReadonlyArray1<f64>
    ) -> Self {
        PyInvestProd(DefaultProd::new(
            a.as_array().to_owned(),
            alpha.as_array().to_owned(),
            b.as_array().to_owned(),
            beta.as_array().to_owned(),
        ).expect("invalid production function parameters"))
    }

    fn f_i(&self, i: usize, actions: &PyInvestActions) -> (f64, f64) {
        self.0.f_i(i, &actions.0)
    }

    fn f<'py>(&self, py: Python<'py>, actions: &PyInvestActions) -> (&'py PyArray1<f64>, &'py PyArray1<f64>) {
        let (s, p) = self.0.f(&actions.0);
        (s.into_pyarray(py), p.into_pyarray(py))
    }
}

// create python class container "LinearReward" for LinearReward

#[pyclass(name = "LinearReward")]
#[derive(Clone)]
pub struct PyLinearReward(LinearReward);

#[pymethods]
impl PyLinearReward {
    #[new]
    fn new(
        win_a: PyReadonlyArray1<f64>, win_b: PyReadonlyArray1<f64>,
        lose_a: PyReadonlyArray1<f64>, lose_b: PyReadonlyArray1<f64>
    ) -> Self {
        PyLinearReward(LinearReward::new(
            win_a.as_array().to_owned(),
            win_b.as_array().to_owned(),
            lose_a.as_array().to_owned(),
            lose_b.as_array().to_owned(),
        ).expect("invalid reward function parameters"))
    }
}

// create python class container "PayoffFunc" for DefaultPayoff

type DefaultPayoff_ = DefaultPayoff<
    Actions,
    DefaultProd<Actions>,
    WinnerOnlyRisk,
    DefaultCSF,
    LinearReward,
    ConstantDisasterCost,
    FixedUnitCost
>;

#[derive(Clone)]
#[pyclass(name = "PayoffFunc")]
pub struct PyDefaultPayoff(DefaultPayoff_);

#[pymethods]
impl PyDefaultPayoff {
    #[new]
    pub fn new(
        prod_func: PyDefaultProd,
        reward_func: PyLinearReward,
        theta: PyReadonlyArray1<f64>,
        d: PyReadonlyArray1<f64>,
        r: PyReadonlyArray1<f64>,
    ) -> Self {
        PyDefaultPayoff(DefaultPayoff::new(
            prod_func.0,
            WinnerOnlyRisk { theta: theta.as_array().to_owned() },
            DefaultCSF,
            reward_func.0,
            ConstantDisasterCost { d: d.as_array().to_owned() },
            FixedUnitCost { r: r.as_array().to_owned() },
        ).expect("invalid payoff function parameters"))
    }

    pub fn u_i(&self, i: usize, actions: &PyActions) -> f64 {
        self.0.u_i(i, &actions.0)
    }

    pub fn u<'py>(&self, py: Python<'py>, actions: &PyActions) -> &'py PyArray1<f64> {
        self.0.u(&actions.0).into_pyarray(py)
    }
}

type InvestPayoff_ = DefaultPayoff<
    InvestActions,
    DefaultProd<InvestActions>,
    WinnerOnlyRisk,
    DefaultCSF,
    LinearReward,
    ConstantDisasterCost,
    FixedInvestCost
>;

#[derive(Clone)]
#[pyclass(name = "InvestPayoffFunc")]
pub struct PyInvestPayoff(InvestPayoff_);

#[pymethods]
impl PyInvestPayoff {
    #[new]
    pub fn new(
        prod_func: PyInvestProd,
        reward_func: PyLinearReward,
        theta: PyReadonlyArray1<f64>,
        d: PyReadonlyArray1<f64>,
        r_x: PyReadonlyArray1<f64>,
        r_inv: PyReadonlyArray1<f64>,
    ) -> Self {
        PyInvestPayoff(DefaultPayoff::new(
            prod_func.0,
            WinnerOnlyRisk { theta: theta.as_array().to_owned() },
            DefaultCSF,
            reward_func.0,
            ConstantDisasterCost { d: d.as_array().to_owned() },
            FixedInvestCost::new(
                r_x.as_array().to_owned(),
                r_inv.as_array().to_owned(),
            ),
        ).expect("invalid payoff function parameters"))
    }

    pub fn u_i(&self, i: usize, actions: &PyInvestActions) -> f64 {
        self.0.u_i(i, &actions.0)
    }

    pub fn u<'py>(&self, py: Python<'py>, actions: &PyInvestActions) -> &'py PyArray1<f64> {
        self.0.u(&actions.0).into_pyarray(py)
    }
}

// create python class container for SolverOptions

#[pyclass(name = "SolverOptions")]
pub struct PySolverOptions {
    max_iters: u64,
    tol: f64,
    init_simplex_size: f64,
    nm_max_iters: u64,
    nm_tol: f64,
}

#[pymethods]
impl PySolverOptions {
    #[new]
    #[args(
        max_iters = "100",
        tol = "1e-6",
        init_simplex_size = "1.0",
        nm_max_iters = "100",
        nm_tol = "1e-8",
    )]
    pub fn new(
        max_iters: u64,
        tol: f64,
        init_simplex_size: f64,
        nm_max_iters: u64,
        nm_tol: f64,
    ) -> Self {
        PySolverOptions { max_iters, tol, init_simplex_size, nm_max_iters, nm_tol }
    }
}

// create python class container "Aggregator" for ExponentialDiscounter

#[pyclass(name = "Aggregator")]
pub struct PyExponentialDiscounter(ExponentialDiscounter<DefaultPayoff_, DefaultPayoff_>);

#[pymethods]
impl PyExponentialDiscounter {
    #[new]
    pub fn new(states: &PyList, gammas: Vec<f64>) -> Self {
        let states_vec = states.iter().map(|s|
            s.extract::<PyDefaultPayoff>()
             .expect("states should contain only objects of type PyDefaultPayoff").0
        ).collect();
        PyExponentialDiscounter(ExponentialDiscounter::new(states_vec, gammas))
    }

    pub fn u_i(&self, i: usize, strategies: &PyStrategies) -> f64 {
        self.0.u_i(i, &strategies.0)
    }

    pub fn u<'py>(&self, py: Python<'py>, strategies: &PyStrategies) -> &'py PyArray1<f64> {
        self.0.u(&strategies.0).into_pyarray(py)
    }

    pub fn solve(&self, init_guess: PyStrategies, options: &PySolverOptions) -> PyResult<PyStrategies> {
        let solver_options = SolverOptions {
            init_guess: init_guess.0,
            max_iters: options.max_iters,
            tol: options.tol,
            nm_options: NMOptions {
                init_simplex_size: options.init_simplex_size,
                max_iters: options.nm_max_iters,
                tol: options.nm_tol,
            }
        };
        let res = solve(&self.0, &solver_options);
        match res {
            Ok(res) => Ok(PyStrategies(res)),
            Err(e) => Err(PyException::new_err(format!("{}", e))),
        }
    }
}

#[pyclass(name = "InvestAggregator")]
pub struct PyInvestExpDiscounter(InvestExponentialDiscounter<InvestPayoff_>);

#[pymethods]
impl PyInvestExpDiscounter {
    #[new]
    pub fn new(state0: PyInvestPayoff, gammas: Vec<f64>) -> Self {
        PyInvestExpDiscounter(InvestExponentialDiscounter::new(state0.0, gammas))
    }

    pub fn u_i(&self, i: usize, strategies: &PyInvestStrategies) -> f64 {
        self.0.u_i(i, &strategies.0)
    }

    pub fn u<'py>(&self, py: Python<'py>, strategies: &PyInvestStrategies) -> &'py PyArray1<f64> {
        self.0.u(&strategies.0).into_pyarray(py)
    }

    pub fn solve(&self, init_guess: PyInvestStrategies, options: &PySolverOptions) -> PyResult<PyInvestStrategies> {
        let solver_options = SolverOptions {
            init_guess: init_guess.0,
            max_iters: options.max_iters,
            tol: options.tol,
            nm_options: NMOptions {
                init_simplex_size: options.init_simplex_size,
                max_iters: options.nm_max_iters,
                tol: options.nm_tol,
            }
        };
        let res = solve(&self.0, &solver_options);
        match res {
            Ok(res) => Ok(PyInvestStrategies(res)),
            Err(e) => Err(PyException::new_err(format!("{}", e))),
        }
    }
}
