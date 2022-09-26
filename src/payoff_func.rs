use crate::prod_func::ProdFunc;
use crate::risk_func::RiskFunc;
use crate::csf::CSF;
use crate::reward_func::RewardFunc;
use crate::cost_func::CostFunc;
use crate::disaster_cost::DisasterCost;

pub trait PayoffFunc {
    fn u_i(&self, i: usize, s: &Vec<f64>, p: &Vec<f64>) -> f64;
    fn u(&self, s: &Vec<f64>, p: &Vec<f64>) -> Vec<f64> {
        (0..p.len()).map(|i| self.u_i(i, s, p)).collect()
    }
}

pub struct DefaultPayoff<T, U, V, W, X, Y>
where T: ProdFunc,
      U: RiskFunc,
      V: CSF,
      W: RewardFunc,
      X: DisasterCost,
      Y: CostFunc
{
    pub prod_func: T,
    pub risk_func: U,
    pub csf: V,
    pub reward_func: W,
    pub disaster_cost: X,
    pub cost_func: Y,
}

impl<T, U, V, W, X, Y> PayoffFunc for DefaultPayoff<T, U, V, W, X, Y>
where T: ProdFunc,
      U: RiskFunc,
      V: CSF,
      W: RewardFunc,
      X: DisasterCost,
      Y: CostFunc
{
    fn u_i(&self, i: usize, xs: &Vec<f64>, xp: &Vec<f64>) -> f64 {
        let (s, p) = self.prod_func.f(xs, xp);

        let sigmas = self.risk_func.sigma(&s, &p);
        let qs = self.csf.q(&p);
        let rewards = self.reward_func.reward(i, &p);
        // payoff given no disaster * proba no disaster
        let no_d = sigmas.iter().zip(qs.iter()).zip(rewards.iter()).map(
            |((sigma, q), reward)| sigma * q * reward
        ).sum::<f64>();
        // cost given disaster * proba disaster
        let yes_d = (1.0 - sigmas.iter().zip(qs.iter()).map(
            |(sigma, q)| sigma * q
        ).sum::<f64>()) * self.disaster_cost.d_i(i, &s, &p);

        no_d - yes_d - self.cost_func.c_i(i, &xs, &xp)
    }
}