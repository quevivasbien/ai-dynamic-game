import numpy as np
import dynapai as dp

from time import time

prodFunc = dp.ProdFunc(
    a = np.array([10., 10.]),
    alpha = np.array([0.5, 0.5]),
    b = np.array([10., 10.]),
    beta = np.array([0.5, 0.5])
)

rewardFunc = dp.LinearReward(np.ones(2), np.zeros(2), np.zeros(2), np.zeros(2))

payoffFunc = dp.PayoffFunc(
    prod_func = prodFunc,
    reward_func = rewardFunc,
    theta = np.array([0.5, 0.5]),
    d = np.array([1.0, 1.0]),
    r = np.array([0.1, 0.1])
)

actions = dp.Actions(
    xs = np.array([1., 1.]),
    xp = np.array([2., 2.])
)

print("Actions:", actions)
print("Payoff from actions:", payoffFunc.u(actions))

strategies = dp.Strategies.from_actions([actions, actions, actions])

print("Strategies:", strategies)

agg = dp.Aggregator(
    states = [payoffFunc, payoffFunc, payoffFunc],
    gammas = [0.9, 0.5]
)

print("Aggregate payoff from strategies:", agg.u(strategies))

solverOptions = dp.SolverOptions(
    max_iters = 100,
    tol = 1e-6,
    init_simplex_size = 1.,
    nm_max_iters = 100,
    nm_tol = 1e-8,
)

time0 = time()
res = agg.solve(init_guess = strategies, options = solverOptions)
time1 = time()
print(f"Solved in {time1 - time0:.3f} seconds")
print("Optimal strategies:", res, sep = '\n')

payoffFunc = dp.InvestPayoffFunc(
    prod_func = prodFunc.with_invest(),
    reward_func = rewardFunc,
    theta = np.array([0.5, 0.5]),
    d = np.array([1.0, 1.0]),
    r_x = np.array([0.1, 0.1]),
    r_inv = np.array([0.01, 0.01])
)

actions = dp.InvestActions(
    xs = np.array([1., 1.]),
    xp = np.array([2., 2.]),
    inv_s = np.array([0.5, 0.5]),
    inv_p = np.array([0.1, 0.1])
)
print("Invest actions:", actions)

strategies = dp.InvestStrategies.from_actions([actions, actions, actions])
print("Invest strategies:", strategies)

agg = dp.InvestAggregator(
    state0 = payoffFunc,
    gammas = [0.9, 0.5]
)

print("Aggregate payoff from invest strategies:", agg.u(strategies))

time0 = time()
res = agg.solve(init_guess = strategies, options = solverOptions)
time1 = time()
print(f"Solved in {time1 - time0:.3f} seconds")
print("Optimal invest strategies:", res, sep = '\n')
