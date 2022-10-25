use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray3, IntoPyArray};
use numpy::ndarray::{Array1};
use pyo3::exceptions::PyException;
use pyo3::{prelude::*, types::PyList};

use crate::cost_func::{FixedUnitCost, FixedInvestCost};
use crate::csf::DefaultCSF;
use crate::disaster_cost::ConstantDisasterCost;
use crate::payoff_func::{PayoffFunc, DefaultPayoff};
use crate::prod_func::{ProdFunc, DefaultProd};
use crate::reward_func::LinearReward;
use crate::risk_func::WinnerOnlyRisk;
use crate::scenarios::Scenario;
use crate::solve::{InitGuess, NMOptions, SolverOptions, solve};
use crate::states::{PayoffAggregator, ExponentialDiscounter, InvestExponentialDiscounter};
use crate::strategies::*;
use crate::init_rep;


// implement python class containers for Actions and Strategies

#[derive(Clone)]
#[pyclass(name = "Actions")]
pub struct PyActions(Actions);

#[pymethods]
impl PyActions {
    #[new]
    fn from_inputs(xs: PyReadonlyArray1<f64>, xp: PyReadonlyArray1<f64>) -> Self {
        PyActions(
            match Actions::from_inputs(xs.as_array().to_owned(), xp.as_array().to_owned()) {
                Ok(actions) => actions,
                Err(e) => panic!("{}", e),
            }
        )
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
        PyInvestActions(
            match InvestActions::from_inputs(
                xs.as_array().to_owned(), xp.as_array().to_owned(),
                inv_s.as_array().to_owned(), inv_p.as_array().to_owned(),
            ) {
                Ok(actions) => actions,
                Err(e) => panic!("{}", e),
            }
        )
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
        PyStrategies(match Strategies::from_array(x.as_array().to_owned()) {
            Ok(strategies) => strategies,
            Err(e) => panic!("{}", e),
        })
    }

    #[staticmethod]
    fn from_actions(actions: &PyList) -> Self {
        let actions_vec = actions.iter().map(|a|
            a.extract::<PyActions>()
             .expect("actions should contain only objects of type PyActions").0
        ).collect();
        PyStrategies(
            match Strategies::from_actions(actions_vec) {
                Ok(strategies) => strategies,
                Err(e) => panic!("{}", e),
            }
        )
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
        PyInvestStrategies(match InvestStrategies::from_array(x.as_array().to_owned()) {
            Ok(strategies) => strategies,
            Err(e) => panic!("{}", e),
        })
    }

    #[staticmethod]
    fn from_actions(actions: &PyList) -> Self {
        let actions_vec = actions.iter().map(|a|
            a.extract::<PyInvestActions>()
             .expect("actions should contain only objects of type PyInvestActions").0
        ).collect();
        PyInvestStrategies(
            match InvestStrategies::from_actions(actions_vec) {
                Ok(strategies) => strategies,
                Err(e) => panic!("{}", e),
            }
        )
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
        PyDefaultProd(DefaultProd::new(
            a.as_array().to_owned(),
            alpha.as_array().to_owned(),
            b.as_array().to_owned(),
            beta.as_array().to_owned(),
        ).expect("invalid production function parameters"))
    }

    #[staticmethod]
    fn expand_from<'py>(
        py: Python<'py>,
        a: &PyList, alpha: &PyList,
        b: &PyList, beta: &PyList,
    ) -> &'py PyList {
        let a_vec = a.iter().map(|a|
            a.extract::<PyReadonlyArray1<f64>>().unwrap().as_array().to_owned()
        ).collect();
        let alpha_vec = alpha.iter().map(|a|
            a.extract::<PyReadonlyArray1<f64>>().unwrap().as_array().to_owned()
        ).collect();
        let b_vec = b.iter().map(|a|
            a.extract::<PyReadonlyArray1<f64>>().unwrap().as_array().to_owned()
        ).collect();
        let beta_vec = beta.iter().map(|a|
            a.extract::<PyReadonlyArray1<f64>>().unwrap().as_array().to_owned()
        ).collect();
        let prod_funcs = init_rep!(DefaultProd =>
            a: Array1<f64> = a_vec;
            alpha: Array1<f64> = alpha_vec;
            b: Array1<f64> = b_vec;
            beta: Array1<f64> = beta_vec
        );
        PyList::new(
            py,
            prod_funcs.into_iter().map(|p| PyDefaultProd(p).into_py(py))
        )
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

    fn __str__(&self) -> String {
        format!("{}", self.0)
    }
}

#[pyclass(name = "InvestProdFunc")]
#[derive(Clone)]
pub struct PyInvestProd(DefaultProd);

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

    #[staticmethod]
    fn expand_from<'py>(
        py: Python<'py>,
        a: &PyList, alpha: &PyList,
        b: &PyList, beta: &PyList,
    ) -> &'py PyList {
        let a_vec = a.iter().map(|a|
            a.extract::<PyReadonlyArray1<f64>>().unwrap().as_array().to_owned()
        ).collect();
        let alpha_vec = alpha.iter().map(|a|
            a.extract::<PyReadonlyArray1<f64>>().unwrap().as_array().to_owned()
        ).collect();
        let b_vec = b.iter().map(|a|
            a.extract::<PyReadonlyArray1<f64>>().unwrap().as_array().to_owned()
        ).collect();
        let beta_vec = beta.iter().map(|a|
            a.extract::<PyReadonlyArray1<f64>>().unwrap().as_array().to_owned()
        ).collect();
        let prod_funcs = init_rep!(DefaultProd =>
            a: Array1<f64> = a_vec;
            alpha: Array1<f64> = alpha_vec;
            b: Array1<f64> = b_vec;
            beta: Array1<f64> = beta_vec
        );
        PyList::new(
            py,
            prod_funcs.into_iter().map(|p| PyInvestProd(p).into_py(py))
        )
    }

    fn f_i(&self, i: usize, actions: &PyInvestActions) -> (f64, f64) {
        self.0.f_i(i, &actions.0)
    }

    fn f<'py>(&self, py: Python<'py>, actions: &PyInvestActions) -> (&'py PyArray1<f64>, &'py PyArray1<f64>) {
        let (s, p) = self.0.f(&actions.0);
        (s.into_pyarray(py), p.into_pyarray(py))
    }

    fn __str__(&self) -> String {
        format!("{}", self.0)
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

    fn __str__(&self) -> String {
        format!("{:?}", self.0)
    }
}

// create python class container "PayoffFunc" for DefaultPayoff

type DefaultPayoff_ = DefaultPayoff<
    Actions,
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
    fn new(
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

    #[staticmethod]
    pub fn expand_from<'py>(
        py: Python<'py>,
        prod_func_list: &PyList,
        reward_func_list: &PyList,
        theta_list: &PyList,
        d_list: &PyList,
        r_list: &PyList,
    ) -> &'py PyList {
        let prod_funcs = prod_func_list.iter().map(|x|
            x.extract::<PyDefaultProd>().unwrap().0
        ).collect::<Vec<_>>();
        let reward_funcs = reward_func_list.iter().map(|x|
            x.extract::<PyLinearReward>().unwrap().0
        ).collect::<Vec<_>>();
        let risk_funcs = theta_list.iter().map(|x|
            WinnerOnlyRisk { theta: x.extract::<PyReadonlyArray1<f64>>().unwrap().as_array().to_owned() }
        ).collect::<Vec<_>>();
        let disaster_costs = d_list.iter().map(|x|
            ConstantDisasterCost { d: x.extract::<PyReadonlyArray1<f64>>().unwrap().as_array().to_owned() }
        ).collect::<Vec<_>>();
        let cost_funcs = r_list.iter().map(|r| {
            FixedUnitCost { r: r.extract::<PyReadonlyArray1<f64>>().unwrap().as_array().to_owned() }
        }).collect::<Vec<_>>();

        let payoff_funcs = init_rep!(DefaultPayoff_ =>
            prod_func: DefaultProd = prod_funcs;
            risk_func: WinnerOnlyRisk = risk_funcs;
            csf: DefaultCSF = vec![DefaultCSF];
            reward_func: LinearReward = reward_funcs;
            disaster_cost: ConstantDisasterCost = disaster_costs;
            cost_funcs: FixedUnitCost = cost_funcs
        );

        PyList::new(
            py,
            payoff_funcs.into_iter().map(|x|
                PyDefaultPayoff(x).into_py(py)
            )
        )
    }

    fn u_i(&self, i: usize, actions: &PyActions) -> f64 {
        self.0.u_i(i, &actions.0)
    }

    fn u<'py>(&self, py: Python<'py>, actions: &PyActions) -> &'py PyArray1<f64> {
        self.0.u(&actions.0).into_pyarray(py)
    }

    fn __str__(&self) -> String {
        format!(
            "PayoffFunc (PyDefaultPayoff):\nprod_func = {}\nreward_func = {}\ntheta = {}\nd = {}\nr = {}",
            self.0.prod_func,
            self.0.reward_func,
            self.0.risk_func.theta,
            self.0.disaster_cost.d,
            self.0.cost_func.r
        )
    } 
}

type InvestPayoff_ = DefaultPayoff<
    InvestActions,
    DefaultProd,
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

    #[staticmethod]
    pub fn expand_from<'py>(
        py: Python<'py>,
        prod_func_list: &PyList,
        reward_func_list: &PyList,
        theta_list: &PyList,
        d_list: &PyList,
        r_x_list: &PyList,
        r_inv_list: &PyList,
    ) -> &'py PyList {
        let prod_funcs = prod_func_list.iter().map(|x|
            x.extract::<PyInvestProd>().unwrap().0
        ).collect::<Vec<_>>();
        let reward_funcs = reward_func_list.iter().map(|x|
            x.extract::<PyLinearReward>().unwrap().0
        ).collect::<Vec<_>>();
        let risk_funcs = theta_list.iter().map(|x|
            WinnerOnlyRisk { theta: x.extract::<PyReadonlyArray1<f64>>().unwrap().as_array().to_owned() }
        ).collect::<Vec<_>>();
        let disaster_costs = d_list.iter().map(|x|
            ConstantDisasterCost { d: x.extract::<PyReadonlyArray1<f64>>().unwrap().as_array().to_owned() }
        ).collect::<Vec<_>>();
        let cost_funcs = r_x_list.iter().zip(r_inv_list.iter()).map(|(r_x, r_inv)| {
            FixedInvestCost::new(
                r_x.extract::<PyReadonlyArray1<f64>>().unwrap().as_array().to_owned(),
                r_inv.extract::<PyReadonlyArray1<f64>>().unwrap().as_array().to_owned(),
            )
        }).collect::<Vec<_>>();

        let payoff_funcs = init_rep!(InvestPayoff_ =>
            prod_func: DefaultProd = prod_funcs;
            risk_func: WinnerOnlyRisk = risk_funcs;
            csf: DefaultCSF = vec![DefaultCSF];
            reward_func: LinearReward = reward_funcs;
            disaster_cost: ConstantDisasterCost = disaster_costs;
            cost_funcs: FixedInvestCost = cost_funcs
        );

        PyList::new(
            py,
            payoff_funcs.into_iter().map(|x|
                PyInvestPayoff(x).into_py(py)
            )
        )
    }

    pub fn u_i(&self, i: usize, actions: &PyInvestActions) -> f64 {
        self.0.u_i(i, &actions.0)
    }

    pub fn u<'py>(&self, py: Python<'py>, actions: &PyInvestActions) -> &'py PyArray1<f64> {
        self.0.u(&actions.0).into_pyarray(py)
    }

    fn __str__(&self) -> String {
        format!(
            "InvestPayoffFunc (PyInvestPayoff):\nprod_func = {}\nreward_func = {}\ntheta = {}\nd = {}\nr_x = {}\nr_inv = {}",
            self.0.prod_func,
            self.0.reward_func,
            self.0.risk_func.theta,
            self.0.disaster_cost.d,
            self.0.cost_func.r_x,
            self.0.cost_func.r_inv,
        )
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
    fn new(
        max_iters: u64,
        tol: f64,
        init_simplex_size: f64,
        nm_max_iters: u64,
        nm_tol: f64,
    ) -> Self {
        PySolverOptions { max_iters, tol, init_simplex_size, nm_max_iters, nm_tol }
    }

    fn __str__(&self) -> String {
        format!(
            "SolverOptions:\nmax_iters = {}\ntol = {}\ninit_simplex_size = {}\nnm_max_iters = {}\nnm_tol = {}",
            self.max_iters, self.tol, self.init_simplex_size, self.nm_max_iters, self.nm_tol
        )
    }
}

fn expand_options<S: StrategyType>(init_guess: InitGuess<S>, options: &PySolverOptions) -> SolverOptions<S> {
    SolverOptions {
        init_guess: init_guess,
        max_iters: options.max_iters,
        tol: options.tol,
        nm_options: NMOptions {
            init_simplex_size: options.init_simplex_size,
            max_iters: options.nm_max_iters,
            tol: options.nm_tol,
        }
    }
}

type ExpDiscounter_ = ExponentialDiscounter<DefaultPayoff_, DefaultPayoff_>;

// create python class container "Aggregator" for ExponentialDiscounter
#[derive(Clone)]
#[pyclass(name = "Aggregator")]
pub struct PyExponentialDiscounter(ExpDiscounter_);

fn extract_init(init: &PyAny, n: usize) -> PyResult<InitGuess<Strategies>> {
    match init.extract::<PyStrategies>() {
        Ok(s) => Ok(InitGuess::Fixed(s.0)),
        Err(_) => Ok(InitGuess::Random(n)),
    }
}

#[pymethods]
impl PyExponentialDiscounter {
    #[new]
    fn new(states: &PyList, gammas: PyReadonlyArray1<f64>) -> Self {
        let states_vec = states.iter().map(|s|
            s.extract::<PyDefaultPayoff>()
             .expect("states should contain only objects of type PyDefaultPayoff").0
        ).collect();
        PyExponentialDiscounter(
            match ExponentialDiscounter::new(states_vec, gammas.as_array().to_owned()) {
                Ok(discounter) => discounter,
                Err(e) => panic!("Error when constructing aggregator: {}", e),
            }
        )
    }

    #[staticmethod]
    fn expand_from<'py>(
        py: Python<'py>,
        states_list: &PyList,
        gammas_list: &PyList,
    ) -> &'py PyList {
        let states_vec_vec = states_list.iter().map(|s|
            s.extract::<Vec<PyDefaultPayoff>>()
             .unwrap().into_iter().map(|x| x.0).collect()
        ).collect::<Vec<Vec<_>>>();
        let gammas_vec = gammas_list.iter().map(|g|
            g.extract::<PyReadonlyArray1<f64>>().unwrap().as_array().to_owned()
        ).collect::<Vec<_>>();
        let aggs = init_rep!(ExpDiscounter_ =>
            states: Vec<DefaultPayoff_> = states_vec_vec;
            gammas: Array1<f64> = gammas_vec
        );
        PyList::new(
            py,
            aggs.into_iter().map(|agg|
                PyExponentialDiscounter(agg).into_py(py)
            )
        )
    }

    fn u_i(&self, i: usize, strategies: &PyStrategies) -> f64 {
        self.0.u_i(i, &strategies.0)
    }

    fn u<'py>(&self, py: Python<'py>, strategies: &PyStrategies) -> &'py PyArray1<f64> {
        self.0.u(&strategies.0).into_pyarray(py)
    }

    fn solve(&self, init: &PyAny, options: &PySolverOptions) -> PyResult<PyStrategies> {
        let init_guess: InitGuess<Strategies> = extract_init(init, self.0.states.len())?;
        let solver_options = expand_options(init_guess, &options);
        let res = solve(&self.0, &solver_options);
        match res {
            Ok(res) => Ok(PyStrategies(res)),
            Err(e) => Err(PyException::new_err(format!("{}", e))),
        }
    }
}

type InvestExpDiscounter_ = InvestExponentialDiscounter<InvestPayoff_>;

#[derive(Clone)]
#[pyclass(name = "InvestAggregator")]
pub struct PyInvestExpDiscounter(InvestExpDiscounter_);

fn extract_invest_init(init: &PyAny) -> PyResult<InitGuess<InvestStrategies>> {
    match init.extract::<PyInvestStrategies>() {
        Ok(s) => Ok(InitGuess::Fixed(s.0)),
        Err(_) => match init.extract::<usize>() {
            Ok(n) => Ok(InitGuess::Random(n)),
            Err(_) => Err(PyException::new_err("init must be either an InvestStrategies object or a positive integer")),
        }
    }
}

#[pymethods]
impl PyInvestExpDiscounter {
    #[new]
    fn new(state0: PyInvestPayoff, gammas: PyReadonlyArray1<f64>) -> Self {
        PyInvestExpDiscounter(InvestExponentialDiscounter::new(state0.0, gammas.as_array().to_owned()).unwrap())
    }

    #[staticmethod]
    fn expand_from<'py>(
        py: Python<'py>,
        state0_list: &PyList,
        gammas_list: &PyList,
    ) -> &'py PyList {
        let state0_vec = state0_list.iter().map(|s|
            s.extract::<PyInvestPayoff>()
             .unwrap().0
        ).collect::<Vec<_>>();
        let gammas_vec = gammas_list.iter().map(|g|
            g.extract::<PyReadonlyArray1<f64>>().unwrap().as_array().to_owned()
        ).collect::<Vec<_>>();
        let aggs = init_rep!(InvestExpDiscounter_ =>
            state0: InvestPayoff_ = state0_vec;
            gammas: Array1<f64> = gammas_vec
        );
        PyList::new(
            py,
            aggs.into_iter().map(|agg|
                PyInvestExpDiscounter(agg).into_py(py)
            )
        )
    }

    fn u_i(&self, i: usize, strategies: &PyInvestStrategies) -> f64 {
        self.0.u_i(i, &strategies.0)
    }

    fn u<'py>(&self, py: Python<'py>, strategies: &PyInvestStrategies) -> &'py PyArray1<f64> {
        self.0.u(&strategies.0).into_pyarray(py)
    }

    fn solve(&self, init: &PyAny, options: &PySolverOptions) -> PyResult<PyInvestStrategies> {
        let init_guess = extract_invest_init(init)?;
        let solver_options = expand_options(init_guess, &options);
        let res = solve(&self.0, &solver_options);
        match res {
            Ok(res) => Ok(PyInvestStrategies(res)),
            Err(e) => Err(PyException::new_err(format!("{}", e))),
        }
    }
}

#[pyclass(name = "Scenario")]
pub struct PyScenario(Scenario<ExponentialDiscounter<DefaultPayoff_, DefaultPayoff_>>);

#[pymethods]
impl PyScenario {
    #[new]
    fn new(aggs: &PyList) -> PyResult<Self> {
        let aggs_vec = aggs.iter().map(|a|
                a.extract::<PyExponentialDiscounter>()
                .expect("aggs should contain only objects of type PyExponentialDiscounter").0
        ).collect();
        match Scenario::new(aggs_vec) {
            Ok(s) => Ok(PyScenario(s)),
            Err(e) => Err(PyException::new_err(format!("Error when constructing scenario: {}", e))),
        }
    }

    fn solve<'py>(&self, py: Python<'py>, init: &PyAny, options: &PySolverOptions) -> PyResult<&'py PyList> {
        let init_guess: InitGuess<Strategies> = extract_init(init, self.0.aggs()[0].states.len())?;
        let solver_options = expand_options(init_guess, &options);
        match self.0.solve(&solver_options) {
            Ok(res) => {
                let iter = res.into_iter().map(|s|
                    PyCell::new(py, PyStrategies(s)).unwrap()
                );
                Ok(PyList::new(py, iter))
            },
            Err(e) => Err(PyException::new_err(format!("{}", e)))
        }
    }
}

#[pyclass(name = "InvestScenario")]
pub struct PyInvestScenario(Scenario<InvestExponentialDiscounter<InvestPayoff_>>);

#[pymethods]
impl PyInvestScenario {
    #[new]
    fn new(aggs: &PyList) -> PyResult<Self> {
        let aggs_vec = aggs.iter().map(|a|
                a.extract::<PyInvestExpDiscounter>()
                .expect("aggs should contain only objects of type PyInvestExpDiscounter").0
        ).collect();
        match Scenario::new(aggs_vec) {
            Ok(s) => Ok(PyInvestScenario(s)),
            Err(e) => Err(PyException::new_err(format!("Error when constructing scenario: {}", e))),
        }
    }

    fn solve<'py>(&self, py: Python<'py>, init: &PyAny, options: &PySolverOptions) -> PyResult<&'py PyList> {
        let init_guess = extract_invest_init(init)?;
        let solver_options = expand_options(init_guess, &options);
        match self.0.solve(&solver_options) {
            Ok(res) => {
                let iter = res.into_iter().map(|s|
                    PyCell::new(py, PyInvestStrategies(s)).unwrap()
                );
                Ok(PyList::new(py, iter))
            },
            Err(e) => Err(PyException::new_err(format!("{}", e)))
        }
    }
}
