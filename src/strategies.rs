use ndarray::{Array, ArrayView, Axis, Ix2, Ix3, Ix1, stack};

// represents actions for n players in a single time period
#[derive(Clone, Debug)]
pub struct Actions {
    pub n: usize,
    pub x: Array<f64, Ix2>,
}

impl Actions {
    
    pub fn from_inputs(xs: Array<f64, Ix1>, xp: Array<f64, Ix1>) -> Self {
        let x = match stack(Axis(1), &[xs.view(), xp.view()]) {
            Ok(x) => x,
            Err(e) => panic!("Problem creating Actions from (xs, xp): {:?}", e),
        };
        Actions {
            n: x.shape()[0],
            x
        }
    }

    pub fn xs(&self) -> ArrayView<f64, Ix1> {
        self.x.column(0)
    }

    pub fn xp(&self) -> ArrayView<f64, Ix1> {
        self.x.column(1)
    }

}

// represents actions for n players in t time periods
#[derive(Clone, Debug)]
pub struct Strategies {
    pub t: usize,
    pub n: usize,
    pub x: Array<f64, Ix3>,
}

impl Strategies {
    
    pub fn from_array(x: Array<f64, Ix3>) -> Self {
        Strategies { 
            t: x.shape()[0],
            n: x.shape()[1],
            x
        }
    }

    pub fn from_actions(actions: Vec<Actions>) -> Self {
        let t = actions.len();
        let inner_shape = actions[0].x.dim();
        let data: Vec<f64> = actions.into_iter().map(|a| a.x).flatten().collect();
        let x = match Array::from_shape_vec(
            (t, inner_shape.0, inner_shape.1),
            data
        ) {
            Ok(x) => x,
            Err(e) => panic!("Problem creating Strategies from Vec<Actions>: {:?}", e),
        };
        Self::from_array(x)
    }

    pub fn to_actions(self) -> Vec<Actions> {
        self.x.outer_iter().map(move |x| {
            Actions {
                n: x.shape()[0],
                x: x.to_owned(),
            }
        }).collect()
    }

}
