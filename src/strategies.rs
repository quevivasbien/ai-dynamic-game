use ndarray::prelude::*;
use ndarray::{stack, Axis}

// represents actions for n players in a single time period
#[derive(Clone, Debug)]
pub struct Actions {
    pub x: Array<f64, Ix2>,
}

impl Actions {
    pub fn from_inputs(xs: &Array1<f64>, xp: &Array1<f64>) -> Self {
        Actions {
            x: match stack(Axis(0), &[xs.view(), xp.view()]) {
                Ok(x) => x,
                Err(e) => panic!("Problem creating Actions: {:?}", e),
            }
        }
    }
}

// represents actions for n players in t time periods
#[derive(Clone, Debug)]
pub struct Strategies {
    pub x: Array<f64, Ix3>,
}

impl Strategies {
    pub fn from_actions(actions: &Vec<Actions>) -> Self {
        let mut views = Vec::<ArrayView<f64, Ix2>>::new();
        for action in actions {
            views.push(action.x.view());
        }
        Strategies {
            x: stack![Axis(0), views[..]]
        }
    }
}
