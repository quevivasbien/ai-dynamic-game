use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray3, IntoPyArray, PyArray, Ix3};
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
use crate::states::{PayoffAggregator, ExponentialDiscounter, InvestExpDiscounter};
use crate::strategies::*;
use crate::init_rep;


trait PyContainer {
    type Item;
    fn get(&self) -> &Self::Item;
}

// implement python class containers for Actions and Strategies

#[derive(Clone)]
#[pyclass(name = "Actions")]
pub struct PyActions(Actions);

impl PyContainer for PyActions {
    type Item = Actions;
    fn get(&self) -> &Self::Item {
        &self.0
    }
}

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

impl PyContainer for PyInvestActions {
    type Item = InvestActions;
    fn get(&self) -> &Self::Item {
        &self.0
    }
}

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

impl PyContainer for PyStrategies {
    type Item = Strategies;
    fn get(&self) -> &Self::Item {
        &self.0
    }
}

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

    fn data<'py>(&self, py: Python<'py>) -> &'py PyArray<f64, Ix3> {
        self.0.data().to_owned().into_pyarray(py)
    }

    fn __str__(&self) -> String {
        format!("{}", self.0)
    }
}

#[derive(Clone)]
#[pyclass(name = "InvestStrategies")]
pub struct PyInvestStrategies(InvestStrategies);

impl PyContainer for PyInvestStrategies {
    type Item = InvestStrategies;
    fn get(&self) -> &Self::Item {
        &self.0
    }
}

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

    fn data<'py>(&self, py: Python<'py>) -> &'py PyArray<f64, Ix3> {
        self.0.data().to_owned().into_pyarray(py)
    }

    fn __str__(&self) -> String {
        format!("{}", self.0)
    }
}

// create python class container "ProdFunc" for DefaultProd
#[pyclass(name = "ProdFunc")]
#[derive(Clone)]
pub struct PyDefaultProd(DefaultProd);

impl PyContainer for PyDefaultProd {
    type Item = DefaultProd;
    fn get(&self) -> &Self::Item {
        &self.0
    }
}

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
        a: Vec<PyReadonlyArray1<f64>>, alpha: Vec<PyReadonlyArray1<f64>>,
        b: Vec<PyReadonlyArray1<f64>>, beta: Vec<PyReadonlyArray1<f64>>,
    ) -> &'py PyList {
        let a_vec = a.into_iter().map(|x|
            x.as_array().to_owned()
        ).collect();
        let alpha_vec = alpha.into_iter().map(|x|
            x.as_array().to_owned()
        ).collect();
        let b_vec = b.into_iter().map(|x|
            x.as_array().to_owned()
        ).collect();
        let beta_vec = beta.into_iter().map(|x|
            x.as_array().to_owned()
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

impl PyContainer for PyInvestProd {
    type Item = DefaultProd;
    fn get(&self) -> &Self::Item {
        &self.0
    }
}

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
        a: Vec<PyReadonlyArray1<f64>>, alpha: Vec<PyReadonlyArray1<f64>>,
        b: Vec<PyReadonlyArray1<f64>>, beta: Vec<PyReadonlyArray1<f64>>,
    ) -> &'py PyList {
        let a_vec = a.iter().map(|x|
            x.as_array().to_owned()
        ).collect();
        let alpha_vec = alpha.iter().map(|x|
            x.as_array().to_owned()
        ).collect();
        let b_vec = b.iter().map(|x|
            x.as_array().to_owned()
        ).collect();
        let beta_vec = beta.iter().map(|x|
            x.as_array().to_owned()
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

impl PyContainer for PyLinearReward {
    type Item = LinearReward;
    fn get(&self) -> &Self::Item {
        &self.0
    }
}

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

impl PyContainer for PyDefaultPayoff {
    type Item = DefaultPayoff_;
    fn get(&self) -> &Self::Item {
        &self.0
    }
}

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
        prod_func_list: Vec<PyDefaultProd>,
        reward_func_list: Vec<PyLinearReward>,
        theta_list: Vec<PyReadonlyArray1<f64>>,
        d_list: Vec<PyReadonlyArray1<f64>>,
        r_list: Vec<PyReadonlyArray1<f64>>,
    ) -> &'py PyList {
        let prod_funcs = prod_func_list.into_iter().map(|x| x.0).collect::<Vec<_>>();
        let reward_funcs = reward_func_list.into_iter().map(|x| x.0).collect::<Vec<_>>();
        let risk_funcs = theta_list.into_iter().map(|x|
            WinnerOnlyRisk { theta: x.as_array().to_owned() }
        ).collect::<Vec<_>>();
        let disaster_costs = d_list.into_iter().map(|x|
            ConstantDisasterCost { d: x.as_array().to_owned() }
        ).collect::<Vec<_>>();
        let cost_funcs = r_list.into_iter().map(|x| {
            FixedUnitCost { r: x.as_array().to_owned() }
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

impl PyContainer for PyInvestPayoff {
    type Item = InvestPayoff_;
    fn get(&self) -> &Self::Item {
        &self.0
    }
}

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
        prod_func_list: Vec<PyInvestProd>,
        reward_func_list: Vec<PyLinearReward>,
        theta_list: Vec<PyReadonlyArray1<f64>>,
        d_list: Vec<PyReadonlyArray1<f64>>,
        r_x_list: Vec<PyReadonlyArray1<f64>>,
        r_inv_list: Vec<PyReadonlyArray1<f64>>,
    ) -> &'py PyList {
        let prod_funcs = prod_func_list.into_iter().map(|x|
            x.0
        ).collect::<Vec<_>>();
        let reward_funcs = reward_func_list.into_iter().map(|x|
            x.0
        ).collect::<Vec<_>>();
        let risk_funcs = theta_list.into_iter().map(|x|
            WinnerOnlyRisk { theta: x.as_array().to_owned() }
        ).collect::<Vec<_>>();
        let disaster_costs = d_list.into_iter().map(|x|
            ConstantDisasterCost { d: x.as_array().to_owned() }
        ).collect::<Vec<_>>();
        let cost_funcs = r_x_list.into_iter().zip(r_inv_list.into_iter()).map(|(r_x, r_inv)| {
            FixedInvestCost::new(
                r_x.as_array().to_owned(),
                r_inv.as_array().to_owned(),
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
    pub max_iters: u64,
    pub tol: f64,
    pub init_simplex_size: f64,
    pub nm_max_iters: u64,
    pub nm_tol: f64,
}

const DEFAULT_OPTIONS: PySolverOptions = PySolverOptions {
    max_iters: 200,
    tol: 1e-6,
    init_simplex_size: 0.1,
    nm_max_iters: 200,
    nm_tol: 1e-8
};

#[pymethods]
impl PySolverOptions {
    #[new]
    #[args(
        max_iters = "DEFAULT_OPTIONS.max_iters",
        tol = "DEFAULT_OPTIONS.tol",
        init_simplex_size = "DEFAULT_OPTIONS.init_simplex_size",
        nm_max_iters = "DEFAULT_OPTIONS.nm_max_iters",
        nm_tol = "DEFAULT_OPTIONS.nm_tol",
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

impl PyContainer for PyExponentialDiscounter {
    type Item = ExpDiscounter_;
    fn get(&self) -> &Self::Item {
        &self.0
    }
}

fn extract_init<'a, S, P>(init: &'a PyAny) -> PyResult<InitGuess<S>>
where S: StrategyType, P: PyContainer<Item = S> + FromPyObject<'a>
{
    match init.extract::<P>() {
        Ok(s) => Ok(InitGuess::Fixed(s.get().clone())),
        Err(_) => match init.extract::<usize>() {
            Ok(n) => Ok(InitGuess::Random(n)),
            Err(_) => Err(PyException::new_err("init must be either a strategy object or a positive integer"))
        }
    }
}


#[pymethods]
impl PyExponentialDiscounter {
    #[new]
    fn new(state: PyDefaultPayoff, gammas: PyReadonlyArray1<f64>) -> Self {
        PyExponentialDiscounter(
            match ExponentialDiscounter::new(state.0, gammas.as_array().to_owned()) {
                Ok(discounter) => discounter,
                Err(e) => panic!("Error when constructing aggregator: {}", e),
            }
        )
    }

    #[staticmethod]
    fn expand_from<'py>(
        py: Python<'py>,
        states_list: Vec<PyDefaultPayoff>,
        gammas_list: Vec<PyReadonlyArray1<f64>>,
    ) -> &'py PyList {
        let states_vec = states_list.into_iter().map(|x| x.0).collect::<Vec<_>>();
        let gammas_vec = gammas_list.into_iter().map(|x| x.as_array().to_owned()).collect::<Vec<_>>();
        let aggs = init_rep!(ExpDiscounter_ =>
            state: DefaultPayoff_ = states_vec;
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

    #[args(options = "&DEFAULT_OPTIONS")]
    fn solve(&self, init: &PyAny, options: &PySolverOptions) -> PyResult<PyStrategies> {
        let init_guess: InitGuess<Strategies> = extract_init::<_, PyStrategies>(init)?;
        let solver_options = expand_options(init_guess, &options);
        let res = solve(&self.0, &solver_options);
        match res {
            Ok(res) => Ok(PyStrategies(res)),
            Err(e) => Err(PyException::new_err(format!("{}", e))),
        }
    }
}

type InvestExpDiscounter_ = InvestExpDiscounter<InvestPayoff_>;

#[derive(Clone)]
#[pyclass(name = "InvestAggregator")]
pub struct PyInvestExpDiscounter(InvestExpDiscounter_);

impl PyContainer for PyInvestExpDiscounter {
    type Item = InvestExpDiscounter_;
    fn get(&self) -> &Self::Item {
        &self.0
    }
}

#[pymethods]
impl PyInvestExpDiscounter {
    #[new]
    fn new(state0: PyInvestPayoff, gammas: PyReadonlyArray1<f64>) -> Self {
        PyInvestExpDiscounter(InvestExpDiscounter::new(state0.0, gammas.as_array().to_owned()).unwrap())
    }

    #[staticmethod]
    fn expand_from<'py>(
        py: Python<'py>,
        state0_list: Vec<PyInvestPayoff>,
        gammas_list: Vec<PyReadonlyArray1<f64>>,
    ) -> &'py PyList {
        let state0_vec = state0_list.into_iter().map(|x| x.0).collect::<Vec<_>>();
        let gammas_vec = gammas_list.into_iter().map(|x| x.as_array().to_owned()).collect::<Vec<_>>();
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

    #[args(options = "&DEFAULT_OPTIONS")]
    fn solve(&self, init: &PyAny, options: &PySolverOptions) -> PyResult<PyInvestStrategies> {
        let init_guess = extract_init::<_, PyInvestStrategies>(init)?;
        let solver_options = expand_options(init_guess, &options);
        let res = solve(&self.0, &solver_options);
        match res {
            Ok(res) => Ok(PyInvestStrategies(res)),
            Err(e) => Err(PyException::new_err(format!("{}", e))),
        }
    }
}

#[pyclass(name = "Scenario")]
pub struct PyScenario(Scenario<Actions, Strategies, ExponentialDiscounter<DefaultPayoff_, DefaultPayoff_>>);

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

    #[args(options = "&DEFAULT_OPTIONS")]
    fn solve<'py>(&self, py: Python<'py>, init: &PyAny, options: &PySolverOptions) -> PyResult<&'py PyList> {
        let init_guess = extract_init::<_, PyStrategies>(init)?;
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
pub struct PyInvestScenario(Scenario<InvestActions, InvestStrategies, InvestExpDiscounter<InvestPayoff_>>);

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

    #[args(options = "&DEFAULT_OPTIONS")]
    fn solve<'py>(&self, py: Python<'py>, init: &PyAny, options: &PySolverOptions) -> PyResult<&'py PyList> {
        let init_guess = extract_init::<_, PyInvestStrategies>(init)?;
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
