import numpy as np
import dynapai as dp

from time import time

n = 2
t = 5

gammas = np.linspace(0.1, 0.9, n)

prodFunc = dp.ProdFunc(
    a = np.full(n, 10.),
    alpha = np.full(n, 0.5),
    b = np.full(n, 10.),
    beta = np.full(n, 0.5),
)

rewardFunc = dp.LinearReward(np.ones(n), np.zeros(n), np.zeros(n), np.zeros(n))

payoffFunc = dp.PayoffFunc(
    prod_func = prodFunc,
    reward_func = rewardFunc,
    theta = np.full(n, 0.5),
    d = np.full(n, 1.),
    r = np.full(n, 0.1),
)

agg = dp.Aggregator(
    state = payoffFunc,
    gammas = gammas
)

print(f"Solving for {n} players and {t} time steps...")
time0 = time()
res = agg.solve(t)
time1 = time()
print(f"Solved in {time1 - time0:.3f} seconds")
print("Optimal strategies:", res, sep = '\n')
print("Payoff from optimal strategies:", agg.u(res))
print()

dp.plot(res, title = "Optimal strategies")

payoffFunc = dp.InvestPayoffFunc(
    prod_func = prodFunc.with_invest(),
    reward_func = rewardFunc,
    theta = np.full(n, 0.5),
    d = np.full(n, 1.),
    r_x = np.full(n, 0.1),
    r_inv = np.full(n, 0.01),
)

agg = dp.InvestAggregator(
    state0 = payoffFunc,
    gammas = gammas
)

print(f"Solving for {n} players and {t} time steps, with investment...")
time0 = time()
res = agg.solve(t)
time1 = time()
print(f"Solved in {time1 - time0:.3f} seconds")
print("Optimal invest strategies:", res, sep = '\n')
print("Payoff from optimal invest strategies:", agg.u(res))
print()

dp.plot(res, title = "Optimal invest strategies")

# create two prod funcs with different values of theta
payoff_funcs = dp.InvestPayoffFunc.expand_from(
    prod_func_list = [prodFunc.with_invest()],
    reward_func_list = [rewardFunc],
    theta_list = [np.full(n, 0.5), np.full(n, 1.0)],
    d_list = [np.full(n, 1.)],
    r_x_list = [np.full(n, 0.1)],
    r_inv_list = [np.full(n, 0.01)],
)

aggs = dp.InvestAggregator.expand_from(
    state0_list = payoff_funcs,
    gammas_list = [gammas],
)

print("Trying [parallel] solve of scenario...")
scenario = dp.InvestScenario(aggs)
time0 = time()
res = scenario.solve(t)
time1 = time()
print(f"Solved in {time1 - time0:.3f} seconds")
print("Optimal invest strategies:")
for i, r in enumerate(res):
    print(f'Problem {i+1}:\n{r}\n')

