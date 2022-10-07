use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray3, IntoPyArray};
use pyo3::exceptions::PyException;
use pyo3::{prelude::*, types::PyList};

use crate::cost_func::FixedUnitCost;
use crate::csf::DefaultCSF;
use crate::disaster_cost::ConstantDisasterCost;
use crate::payoff_func::{PayoffFunc, DefaultPayoff};
use crate::prod_func::{ProdFunc, DefaultProd};
use crate::reward_func::LinearReward;
use crate::risk_func::WinnerOnlyRisk;
use crate::solve::{NMOptions, SolverOptions, solve};
use crate::states::{PayoffAggregator, ExponentialDiscounter};
use crate::strategies::{Actions, Strategies};


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


// create python class container "ProdFunc" for DefaultProd
#[pyclass(name = "ProdFunc")]
#[derive(Clone)]
pub struct PyDefaultProd(DefaultProd);

#[pymethods]
impl PyDefaultProd {
    #[new]
    fn new(
        a: PyReadonlyArray1<f64>, alpha: PyReadonlyArray1<f64>,
        b: PyReadonlyArray1<f64>, beta: PyReadonlyArray1<f64>
    ) -> Self {
        PyDefaultProd(DefaultProd {
            a: a.as_array().to_owned(),
            alpha: alpha.as_array().to_owned(),
            b: b.as_array().to_owned(),
            beta: beta.as_array().to_owned(),
        })
    }

    fn f_i(&self, i: usize, actions: &PyActions) -> (f64, f64) {
        self.0.f_i(i, &actions.0)
    }

    fn f<'py>(&self, py: Python<'py>, actions: &PyActions) -> (&'py PyArray1<f64>, &'py PyArray1<f64>) {
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
        PyLinearReward(LinearReward {
            win_a: win_a.as_array().to_owned(),
            win_b: win_b.as_array().to_owned(),
            lose_a: lose_a.as_array().to_owned(),
            lose_b: lose_b.as_array().to_owned(),
        })
    }
}

// create python class container "PayoffFunc" for DefaultPayoff

type DefaultPayoff_ = DefaultPayoff<
    DefaultProd,
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
        n: usize,
        prod_func: PyDefaultProd,
        reward_func: PyLinearReward,
        theta: f64,
        d: f64,
        r: f64,
    ) -> Self {
        PyDefaultPayoff(DefaultPayoff {
            prod_func: prod_func.0,
            risk_func: WinnerOnlyRisk::new(n, theta),
            csf: DefaultCSF,
            reward_func: reward_func.0,
            disaster_cost: ConstantDisasterCost::new(n, d),
            cost_func: FixedUnitCost::new(n, r),
        })
    }

    pub fn u_i(&self, i: usize, actions: &PyActions) -> f64 {
        self.0.u_i(i, &actions.0)
    }

    pub fn u<'py>(&self, py: Python<'py>, actions: &PyActions) -> &'py PyArray1<f64> {
        self.0.u(&actions.0).into_pyarray(py)
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
}


// create python class container for SolverOptions

#[pyclass(name = "SolverOptions")]
pub struct PySolverOptions(SolverOptions);

#[pymethods]
impl PySolverOptions {
    #[new]
    pub fn new(
        init_guess: PyStrategies,
        max_iters: u64,
        tol: f64,
        init_simplex_size: f64,
        nm_max_iters: u64,
        nm_tol: f64,
    ) -> Self {
        PySolverOptions(SolverOptions {
            init_guess: init_guess.0,
            max_iters,
            tol,
            nm_options: NMOptions {
                init_simplex_size,
                max_iters: nm_max_iters,
                tol: nm_tol,
            }
        })
    }
}

#[pyfunction(name = "solve")]
pub fn solve_py(
    aggregator: &PyExponentialDiscounter,
    options: &PySolverOptions,
) -> PyResult<PyStrategies> {
    let res = solve(&aggregator.0, &options.0);
    match res {
        Ok(res) => Ok(PyStrategies(res)),
        Err(e) => Err(PyException::new_err(format!("{}", e))),
    }
}
