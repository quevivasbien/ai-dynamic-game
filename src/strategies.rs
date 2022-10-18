use std::fmt;
use numpy::ndarray::{Array, ArrayView, Axis, Ix2, Ix3, Ix1, stack, ArrayViewMut, Slice};

pub trait ActionType: Clone {
    fn n(&self) -> usize;
    fn nparams(&self) -> usize;
    fn data(&self) -> ArrayView<f64, Ix2>;
    fn data_mut(&mut self) -> ArrayViewMut<f64, Ix2>;

    fn from_array(x: Array<f64, Ix2>) -> Self;

    fn xs(&self) -> ArrayView<f64, Ix1>;
    fn xp(&self) -> ArrayView<f64, Ix1>;
}

// represents actions for n players in a single time period
#[derive(Clone, Debug)]
pub struct Actions {
    n: usize,
    x: Array<f64, Ix2>,
}

impl ActionType for Actions {
    fn n(&self) -> usize {
        self.n
    }
    fn nparams(&self) -> usize { 2 }
    fn data(&self) -> ArrayView<f64, Ix2> {
        self.x.view()
    }
    fn data_mut(&mut self) -> ArrayViewMut<f64, Ix2> {
        self.x.view_mut()
    }

    fn from_array(x: Array<f64, Ix2>) -> Self {
        Actions {
            n: x.shape()[0],
            x
        }
    }

    fn xs(&self) -> ArrayView<f64, Ix1> {
        self.x.column(0)
    }
    fn xp(&self) -> ArrayView<f64, Ix1> {
        self.x.column(1)
    }
}

impl Actions {

    pub fn from_inputs(xs: Array<f64, Ix1>, xp: Array<f64, Ix1>) -> Self {
        let x = match stack(Axis(1), &[xs.view(), xp.view()]) {
            Ok(x) => x,
            Err(e) => panic!("Problem creating Actions from (xs, xp): {:?}", e),
        };
        Actions::from_array(x)
    }
}

impl fmt::Display for Actions {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "xs = {}, xp = {}", self.xs(), self.xp())
    }
}

#[derive(Clone, Debug)]
pub struct InvestActions {
    n: usize,
    x: Array<f64, Ix2>,
}

impl ActionType for InvestActions {
    fn n(&self) -> usize {
        self.n
    }
    fn nparams(&self) -> usize { 4 }
    fn data(&self) -> ArrayView<f64, Ix2> {
        self.x.view()
    }
    fn data_mut(&mut self) -> ArrayViewMut<f64, Ix2> {
        self.x.view_mut()
    }

    fn from_array(x: Array<f64, Ix2>) -> Self {
        InvestActions {
            n: x.shape()[0],
            x
        }
    }

    fn xs(&self) -> ArrayView<f64, Ix1> {
        self.x.column(0)
    }
    fn xp(&self) -> ArrayView<f64, Ix1> {
        self.x.column(1)
    }
}

impl InvestActions {

    pub fn from_inputs(
        xs: Array<f64, Ix1>, xp: Array<f64, Ix1>,
        inv_s: Array<f64, Ix1>, inv_p: Array<f64, Ix1>
    ) -> Self {
        let x = match stack(Axis(1), &[xs.view(), xp.view(), inv_s.view(), inv_p.view()]) {
            Ok(x) => x,
            Err(e) => panic!("Problem creating InvestActions from (xs, xp, inv_s, inv_p): {:?}", e),
        };
        InvestActions::from_array(x)
    }

    pub fn inv_s(&self) -> ArrayView<f64, Ix1> {
        self.x.column(2)
    }
    pub fn inv_p(&self) -> ArrayView<f64, Ix1> {
        self.x.column(3)
    }

    // truncate invest data to just contain xs and xp
    pub fn truncate_to_actions(&self) -> Actions {
        Actions::from_array(self.x.slice_axis(Axis(1), Slice::from(0..2)).clone().to_owned())
    }
}

impl fmt::Display for InvestActions {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "xs = {}, xp = {}, inv_s = {}, inv_p = {}", self.xs(), self.xp(), self.inv_s(), self.inv_p())
    }
}

pub trait MutatesOnAction: Clone + Sized {
    type Act: ActionType;

    #[allow(unused_mut)]
    #[allow(unused_variables)]
    fn mutate_on_action_inplace(mut self, actions: &Self::Act) -> Self {
        self
    }
    fn mutate_on_action(&self, actions: &Self::Act) -> Self {
        self.clone().mutate_on_action_inplace(actions)
    }
}

pub trait StrategyType: Clone {
    type Act: ActionType;
    // return number of time periods
    // return number of players
    fn n(&self) -> usize;
    fn t(&self) -> usize;
    fn nparams(&self) -> usize;
    // return n x t x nparams array of params representing strategy
    // where m is the number of parameters for each player + time period
    fn data(&self) -> ArrayView<f64, Ix3>;
    fn data_mut(&mut self) -> ArrayViewMut<f64, Ix3>;

    fn from_array(x: Array<f64, Ix3>) -> Self;
    fn from_actions(actions: Vec<Self::Act>) -> Self {
        let inner_shape = actions[0].data().dim();
        let data = Array::from_shape_fn(
            (inner_shape.0, actions.len(), inner_shape.1),
            |(i, j, k)| actions[j].data()[[i, k]]
        );
        Self::from_array(data)
    }
    fn to_actions(self) -> Vec<Self::Act> {
        self.data().axis_iter(Axis(1)).map(move |x| {
            Self::Act::from_array(x.to_owned())
        }).collect()
    }
}

// represents actions for n players in t time periods
#[derive(Clone, Debug)]
pub struct Strategies {
    n: usize,
    t: usize,
    x: Array<f64, Ix3>,
}

impl StrategyType for Strategies {
    type Act = Actions;

    fn t(&self) -> usize {
        self.t
    }
    fn n(&self) -> usize {
        self.n
    }
    fn nparams(&self) -> usize { 2 }
    fn data(&self) -> ArrayView<f64, Ix3> {
        self.x.view()
    }
    fn data_mut(&mut self) -> ArrayViewMut<f64, Ix3> {
        self.x.view_mut()
    }

    fn from_array(x: Array<f64, Ix3>) -> Self {
        Strategies { 
            n: x.shape()[0],
            t: x.shape()[1],
            x
        }
    }
}


impl fmt::Display for Strategies {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let actions_seq = self.clone().to_actions();
        for (t, actions) in actions_seq.iter().enumerate() {
            write!(f, "t = {}: {}", t, actions)?;
            if t != self.t - 1 {
                write!(f, "\n")?;
            }
        }
        Ok(())
    }
}


#[derive(Clone)]
pub struct InvestStrategies {
    n: usize,
    t: usize,
    x: Array<f64, Ix3>,
}

impl StrategyType for InvestStrategies {

    type Act = InvestActions;

    fn n(&self) -> usize {
        self.n
    }

    fn t(&self) -> usize {
        self.t
    }

    fn nparams(&self) -> usize { 4 }

    fn data(&self) -> ArrayView<f64, Ix3> {
        self.x.view()
    }

    fn data_mut(&mut self) -> ArrayViewMut<f64, Ix3> {
        self.x.view_mut()
    }

    fn from_array(x: Array<f64, Ix3>) -> Self {
        InvestStrategies { 
            n: x.shape()[0],
            t: x.shape()[1],
            x
        }
    }
}

impl fmt::Display for InvestStrategies {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let actions_seq = self.clone().to_actions();
        for (t, actions) in actions_seq.iter().enumerate() {
            write!(f, "t = {}: {}", t, actions)?;
            if t != self.t - 1 {
                write!(f, "\n")?;
            }
        }
        Ok(())
    }
}