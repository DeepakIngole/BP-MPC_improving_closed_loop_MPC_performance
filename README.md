# BP-MPC: Improving Closed-Loop MPC Performance Using BackPropagation

This repository contains the code for the paper **"BP-MPC: Optimizing Closed-Loop MPC Performance Using BackPropagation"**. It provides a Python package and examples for automatically improving Model Predictive Control (MPC) performance using differentiable optimization techniques.

## Features

- Modular implementation of hierarchical MPC with bi-level optimization
- Differentiable QP solver using CasADi
- Symbolic and numerical system dynamics
- Flexible cost and constraint specification
- Support for robust and parallel model scenarios
- Example scripts for cart-pendulum swingup, linear, nonlinear, and random systems
- Utilities for system identification, cost evaluation, and parameter updates

## Installation

Install dependencies:
```bash
pip install -r requirements.txt
```

Clone the repository:
```bash
git clone https://github.com/RiccardoZuliani98/BP-MPC_improving_closed_loop_MPC_performance.git
cd BP-MPC_improving_closed_loop_MPC_performance
```

## Usage

You can run the provided examples to simulate and optimize MPC controllers:

```bash
python examples/example_swingup.py
python examples/example_swingup_parallel_models.py
python examples/random_linear_models/run_models.py
```

Explore the `examples/` folder for more scenarios, including linear, nonlinear, and robust MPC setups.

## Project Structure

- `src/` - Core package modules (dynamics, ingredients, qp, scenario, upper_level, etc.)
- `examples/` - Example scripts and models
- `utils/` - Utility functions (costs, parameter updates, system identification)
- `tests/` - Unit tests for main components

## How It Works

The package formulates MPC as a bi-level optimization problem:
- **Lower Level:** Solves a QP for MPC using symbolic/numeric dynamics and constraints
- **Upper Level:** Optimizes closed-loop performance by updating cost/constraint parameters via gradient-based methods

The workflow is fully differentiable, enabling backpropagation through the closed-loop system and QP solver.

## Example: Cart-Pendulum Swingup

See `examples/example_swingup.py` for a full simulation of a cart-pendulum swingup task with closed-loop optimization.

## License

This project is licensed under the MIT License. See `LICENSE` for details.

## Author

Riccardo Zuliani ([rzuliani@ethz.com](mailto:rzuliani@ethz.com))

## Links

- [Homepage](https://github.com/RiccardoZuliani98/BP-MPC_improving_closed_loop_MPC_performance)
- [Issues](https://github.com/RiccardoZuliani98/BP-MPC_improving_closed_loop_MPC_performance/issues)

