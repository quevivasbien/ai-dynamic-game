use std::fmt;
use numpy::ndarray::{Array, ArrayView, Axis, Ix2, Ix3, Ix1, stack, ArrayViewMut, Slice, ShapeError, s};

pub trait ActionType: Clone + Send + Sync {
    fn data(&self) -> ArrayView<f64, Ix2>;
    fn data_mut(&mut self) -> ArrayViewMut<f64, Ix2>;
    fn n(&self) -> usize { self.data().shape()[0] }
    fn nparams(&self) -> usize { self.data().shape()[1] }

    fn from_array(data: Array<f64, Ix2>) -> Result<Self, String>;

    fn xs(&self) -> ArrayView<f64, Ix1>;
    fn xp(&self) -> ArrayView<f64, Ix1>;
}

// represents actions for n players in a single time period
#[derive(Clone, Debug)]
pub struct Actions(Array<f64, Ix2>);

impl ActionType for Actions {
    fn nparams(&self) -> usize { 2 }

    fn data(&self) -> ArrayView<f64, Ix2> {
        self.0.view()
    }
    fn data_mut(&mut self) -> ArrayViewMut<f64, Ix2> {
        self.0.view_mut()
    }

    fn from_array(x: Array<f64, Ix2>) -> Result<Self, String> {
        if x.shape()[1] != 2 {
            return Err(format!("Actions must have 2 columns, but has {}", x.shape()[1]));
        }
        Ok(Actions(x))
    }

    fn xs(&self) -> ArrayView<f64, Ix1> {
        self.0.slice(s![.., 0])
    }
    fn xp(&self) -> ArrayView<f64, Ix1> {
        self.0.slice(s![.., 1])
    }
}

impl Actions {

    pub fn from_inputs(xs: Array<f64, Ix1>, xp: Array<f64, Ix1>) -> Result<Self, String> {
        let x = match stack(Axis(1), &[xs.view(), xp.view()]) {
            Ok(x) => x,
            Err(e) => return Err(format!("Error when creating Actions from inputs: {}", e))
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
pub struct InvestActions(Array<f64, Ix2>);

impl ActionType for InvestActions {
    fn nparams(&self) -> usize { 4 }
    
    fn data(&self) -> ArrayView<f64, Ix2> {
        self.0.view()
    }
    fn data_mut(&mut self) -> ArrayViewMut<f64, Ix2> {
        self.0.view_mut()
    }

    fn from_array(x: Array<f64, Ix2>) -> Result<Self, String> {
        if x.shape()[1] != 4 {
            return Err(format!("InvestActions must have 4 columns, but has {}", x.shape()[1]));
        }
        Ok(InvestActions(x))
    }

    fn xs(&self) -> ArrayView<f64, Ix1> {
        self.0.slice(s![.., 0])
    }
    fn xp(&self) -> ArrayView<f64, Ix1> {
        self.0.slice(s![.., 1])
    }
}

impl InvestActions {

    pub fn from_inputs(
        xs: Array<f64, Ix1>, xp: Array<f64, Ix1>,
        inv_s: Array<f64, Ix1>, inv_p: Array<f64, Ix1>
    ) -> Result<Self, String> {
        let x = match stack(Axis(1), &[xs.view(), xp.view(), inv_s.view(), inv_p.view()]) {
            Ok(x) => x,
            Err(e) => return Err(format!("Error when creating InvestActions from inputs: {}", e))
        };
        InvestActions::from_array(x)
    }

    pub fn inv_s(&self) -> ArrayView<f64, Ix1> {
        self.0.column(2)
    }
    pub fn inv_p(&self) -> ArrayView<f64, Ix1> {
        self.0.column(3)
    }

    // truncate invest data to just contain xs and xp
    pub fn truncate_to_actions(&self) -> Result<Actions, String> {
        Actions::from_array(self.0.slice_axis(Axis(1), Slice::from(0..2)).clone().to_owned())
    }
}

impl fmt::Display for InvestActions {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "xs = {}, xp = {}, inv_s = {}, inv_p = {}", self.xs(), self.xp(), self.inv_s(), self.inv_p())
    }
}

pub trait MutatesOnAction<A: ActionType>: Clone + Sized {

    #[allow(unused_mut)]
    #[allow(unused_variables)]
    fn mutate_on_action_inplace(mut self, actions: &A) -> Self {
        self
    }
    fn mutate_on_action(&self, actions: &A) -> Self {
        self.clone().mutate_on_action_inplace(actions)
    }
}


pub trait StrategyType: Clone + Send + Sync {
    type Act: ActionType;
    // return n x t x nparams array of params representing strategy
    // where m is the number of parameters for each player + time period
    fn data(&self) -> ArrayView<f64, Ix3>;
    fn data_mut(&mut self) -> ArrayViewMut<f64, Ix3>;

    fn t(&self) -> usize { self.data().shape()[0] }
    fn n(&self) -> usize { self.data().shape()[1] }
    fn nparams(&self) -> usize { self.data().shape()[2] }

    fn from_array(x: Array<f64, Ix3>) -> Self;
    fn from_actions(actions: Vec<Self::Act>) -> Result<Self, ShapeError> {
        let data = stack(
            Axis(0),
            &actions.iter().map(
                |a| a.data()
            ).collect::<Vec<_>>())?;
        Ok(Self::from_array(data))
    }
    
    fn to_actions(self) -> Vec<Self::Act> {
        self.data().outer_iter().map(move |x| {
            Self::Act::from_array(x.to_owned()).unwrap()
        }).collect()
    }
}


// represents actions for n players in t time periods
#[derive(Clone, Debug)]
pub struct Strategies {
    t: usize,
    n: usize,
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
            t: x.shape()[0],
            n: x.shape()[1],
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
    t: usize,
    n: usize,
    x: Array<f64, Ix3>,
}

impl StrategyType for InvestStrategies {
    type Act = InvestActions;

    fn t(&self) -> usize {
        self.t
    }
    fn n(&self) -> usize {
        self.n
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
            t: x.shape()[0],
            n: x.shape()[1],
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