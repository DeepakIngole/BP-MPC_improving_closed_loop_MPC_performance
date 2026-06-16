**Tags:** #class

## Overview

The `Dynamics` class encapsulates a discrete-time dynamical system with a built-in "straight-through" gradient estimator. It allows the framework to seamlessly decouple the forward simulation of a system from its automatic differentiation (AD) paths.

## Purpose

In realistic control scenarios, the actual environment might be non-differentiable, discontinuous, or highly noisy. However, optimization solvers (like MPC) strictly require smooth gradients. This class solves that by holding two separate models:

1. **`true_fun`**: Evaluated exclusively during the primal (forward) pass. It represents the real system or high-fidelity simulator.
    
2. **`nominal_fun`**: Evaluated exclusively during the gradient (backward/JVP) pass. It represents a differentiable surrogate model used to compute Jacobians.
    

When the framework or solver takes a derivative of the `step` or `rollout` methods, JAX automatically routes the math through `nominal_fun` while keeping the forward trajectory rooted in `true_fun`.

## Key Methods

- **`step(state, action, true_params, nominal_params)`**: Advances the system by exactly one time-step. Uses `jax.custom_jvp` under the hood to enforce the straight-through routing.
    
- **`rollout(state0, actions, true_params, nominal_params)`**: Simulates an entire trajectory over a horizon by applying a sequence of actions. It compiles down to a highly efficient `jax.lax.scan` loop.
    

## Dependencies

- [[Variable]]: Used to define `true_params_spec` and `nominal_params_spec`. These define the symbolic descriptors for the parameters that the true and nominal functions expect at runtime.