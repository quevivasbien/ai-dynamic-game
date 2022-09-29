use crate::strategies::Actions;
use crate::prod_func::ProdFunc;
use crate::risk_func::RiskFunc;
use crate::csf::CSF;
use crate::reward_func::RewardFunc;
use crate::cost_func::CostFunc;
use crate::disaster_cost::DisasterCost;

pub trait PayoffFunc {
    fn u_i(&self, i: usize, actions: &Actions) -> f64;
    fn u(&self, actions: &Actions) -> Vec<f64> {
        (0..actions.n).map(|i| self.u_i(i, actions)).collect()
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
    fn u_i(&self, i: usize, actions: &Actions) -> f64 {
        let (s, p) = self.prod_func.f(actions);

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

        no_d - yes_d - self.cost_func.c_i(i, actions)
    }

    fn u(&self, actions: &Actions) -> Vec<f64> {
        let (s, p) = self.prod_func.f(actions);
        let sigmas = self.risk_func.sigma(&s, &p);
        let qs = self.csf.q(&p);

        let all_rewards = (0..p.len()).map(
            |i| self.reward_func.reward(i, &p)
        );

        let no_d = all_rewards.map(
            |rewards| sigmas.iter().zip(qs.iter()).zip(rewards.iter()).map(
                |((sigma, q), reward)| sigma * q * reward
            ).sum::<f64>()
        );

        let proba_d = 1.0 - sigmas.iter().zip(qs.iter()).map(
            |(sigma, q)| sigma * q
        ).sum::<f64>();

        let disaster_costs = self.disaster_cost.d(&s, &p);
        let yes_d = disaster_costs.iter().map(|d| d * proba_d);

        let net_rewards = no_d.zip(yes_d).map(|(n, y)| n - y);

        let cost = self.cost_func.c(actions);

        net_rewards.zip(cost.iter()).map(|(r, c)| r - c).collect()
    }
}
