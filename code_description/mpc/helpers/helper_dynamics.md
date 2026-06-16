**Tags:** #file #helpers

## Overview

The `_dynamics.py` module is responsible for translating system dynamics into the standard equality constraint matrices ($A \cdot z = b$) required by the QP solver. It supports Linear Time-Invariant (LTI), Linear Time-Varying (LTV), and full nonlinear systems.

## Purpose

This file handles the complex indexing required to enforce $x_{k+1} = f(x_k, u_k)$ across the whole decision vector:

1. **Builders**: Construct the block-banded matrices representing the dynamics rollout.
2. **Factories** (`lti_dynamics`, `ltv_dynamics`): Wire up symbolic variables directly into a [[Constraint]] object representing the equality conditions.
3. **Nonlinear Linearization** (`nonlinear_dynamics`): A powerful factory that takes a generic [[Dynamics]] object and nominal reference trajectories. It uses `jax.jacfwd` under the hood to automatically compute the Jacobians ($A_t$, $B_t$) at each timestep, transforming the nonlinear system into an LTV equality constraint that updates dynamically at solve time.

## Dependencies

- [[Variable]]: Used to accept parametric matrices, initial states, or nominal trajectories.
- [[Constraint]]: The factory outputs this object to enforce the dynamics. 
- [[Dynamics]]: Ingested by the `nonlinear_dynamics` factory to automatically compute Jacobians.