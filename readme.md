# ai-dynamic-game

## What is this?

This repository contains code meant to model interactions between actors competing to develop a new, risky technology (we have AI technology in mind in particular, but this could be any technology that carries some risk of unintended negative consequences).

A lot of the functionality of the code here overlaps with [AIIncentives.jl](https://github.com/quevivasbien/AIIncentives.jl/). Both this code and that code are based on the same basic model, though this code specializes in multi-period extensions of that model, while AIIncentives.jl is meant to provide a robust set of tools for studying the static model.

## How to build?

Currently, you need to be able to compile Rust to interact with this project, though I may provide a pre-built Python library in the future.

You can interact directly with the Rust code, or use the provided Python bindings to access some functionality in Python.

### Using the Python bindings

Start by creating a virtual environment in the main directory of this repository: if you have Conda (e.g., Anaconda), you can do that by running
```bash
conda create --name venv python=3.7
```
You can create your venv in some other way and call it whatever you want, but you do need the Python version to be at least 3.7.

You'll then need to activate the virtual environment. With Conda:
```bash
conda activate venv
```

You can then install the maturin build tool in this environment with
```bash
pip install maturin
```

Finally, to compile the Rust code and create a Python library, run
```bash
maturin develop
```

As long as you have the venv you created active, you should then be able to import the Python bindings in a module called `dynapai`; e.g., in Python:
```python
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
    n = 2,
    prod_func = prodFunc,
    reward_func = rewardFunc,
    theta = 0.5,
    d = 1.0,
    r = 0.1
)

actions = dp.Actions(np.array([1., 1.]), np.array([2., 2.]))

print("Payoff from actions:", payoffFunc.u(actions))
```

This should print
```
Payoff from actions: [-0.2099315 -0.2099315]
```
