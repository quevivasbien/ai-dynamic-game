import numpy as np
import dynapai as dp

from time import time

n = 2
t = 10

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

actions = dp.Actions(
    xs = np.full(n, 1.),
    xp = np.full(n, 2.),
)

print("Actions:", actions)
print("Payoff from actions:", payoffFunc.u(actions))

strategies = dp.Strategies.from_actions([actions]*t)

print("Strategies:", strategies, sep = '\n')

agg = dp.Aggregator(
    states = [payoffFunc]*t,
    gammas = gammas
)

print("Aggregate payoff from strategies:", agg.u(strategies))

solverOptions = dp.SolverOptions()

time0 = time()
res = agg.solve(init_guess = strategies, options = solverOptions)
time1 = time()
print(f"Solved in {time1 - time0:.3f} seconds")
print("Optimal strategies:", res, sep = '\n')
print("Payoff from optimal strategies:", agg.u(res))
print()

payoffFunc = dp.InvestPayoffFunc(
    prod_func = prodFunc.with_invest(),
    reward_func = rewardFunc,
    theta = np.full(n, 0.5),
    d = np.full(n, 1.),
    r_x = np.full(n, 0.1),
    r_inv = np.full(n, 0.01),
)

actions = dp.InvestActions(
    xs = np.full(n, 1.),
    xp = np.full(n, 2.),
    inv_s = np.full(n, 10.),
    inv_p = np.full(n, 20.),
)
print("Invest actions:", actions)

strategies = dp.InvestStrategies.from_actions([actions]*t)
print("Invest strategies:", strategies, sep = '\n')

agg = dp.InvestAggregator(
    state0 = payoffFunc,
    gammas = gammas
)

print("Aggregate payoff from invest strategies:", agg.u(strategies))

time0 = time()
res = agg.solve(init_guess = strategies, options = solverOptions)
time1 = time()
print(f"Solved in {time1 - time0:.3f} seconds")
print("Optimal invest strategies:", res, sep = '\n')
print("Payoff from optimal invest strategies:", agg.u(res))
print()

print("Trying [parallel] solve of scenario")
scenario = dp.InvestScenario([agg, agg])
time0 = time()
res = scenario.solve(init_guess = strategies, options = solverOptions)
time1 = time()
print(f"Solved in {time1 - time0:.3f} seconds")
