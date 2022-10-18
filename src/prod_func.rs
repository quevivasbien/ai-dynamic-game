use numpy::ndarray::{Array, Ix1};

use crate::strategies::*;

pub trait ProdFunc: Clone {
    type Act: ActionType;
    fn f_i(&self, i: usize, actions: &Self::Act) -> (f64, f64);
    fn f(&self, actions: &Self::Act) -> (Array<f64, Ix1>, Array<f64, Ix1>) {
        let (s, p) = (0..actions.n()).map(|i| self.f_i(i, actions)).unzip();
        (Array::from_vec(s), Array::from_vec(p))
    }

    fn n(&self) -> usize;
}

#[derive(Clone)]
pub struct DefaultProd<A: ActionType> {
    n: usize,
    pub a: Array<f64, Ix1>,
    pub alpha: Array<f64, Ix1>,
    pub b: Array<f64, Ix1>,
    pub beta: Array<f64, Ix1>,
    phantom: std::marker::PhantomData<A>,
}

impl<A: ActionType> DefaultProd<A> {
    pub fn new(a: Array<f64, Ix1>, alpha: Array<f64, Ix1>, b: Array<f64, Ix1>, beta: Array<f64, Ix1>) -> Result<DefaultProd<A>, &'static str> {
        let n = a.len();
        if n != alpha.len() || n != b.len() || n != beta.len() {
            return Err("When creating new DefaultProd: All input arrays must have the same length");
        }
        Ok(DefaultProd { n, a, alpha, b, beta, phantom: std::marker::PhantomData })
    }
}

impl<A: ActionType> ProdFunc for DefaultProd<A> {
    type Act = A;

    fn f_i(&self, i: usize, actions: &A) -> (f64, f64) {
        (
            self.a[i] * actions.xs()[i].powf(self.alpha[i]),
            self.b[i] * actions.xp()[i].powf(self.beta[i])
        )
    }

    fn n(&self) -> usize {
        self.n
    }
}

impl MutatesOnAction for DefaultProd<InvestActions> {
    type Act = InvestActions;
    fn mutate_on_action_inplace(mut self, actions: &Self::Act) -> Self {
        self.a = self.a + actions.inv_s();
        self.b = self.b + actions.inv_p();
        self
    }
}
