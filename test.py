import numpy as np
import dynapai as dp

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

print("Actions =", actions)
print("Payoff from actions:", payoffFunc.u(actions))

strategies = dp.Strategies.from_actions([actions, actions, actions])

print("Strategies =", strategies)

agg = dp.Aggregator(
    states = [payoffFunc, payoffFunc, payoffFunc],
    gammas = [0.9, 0.5]
)

print("Aggregate payoff from strategies:", agg.u(strategies))

solverOptions = dp.SolverOptions(
    init_guess = strategies,
    max_iters = 100,
    tol = 1e-6,
    init_simplex_size = 1.,
    nm_max_iters = 100,
    nm_tol = 1e-8,
)

print("Optimal strategies:", dp.solve(agg, solverOptions), sep = '\n')
