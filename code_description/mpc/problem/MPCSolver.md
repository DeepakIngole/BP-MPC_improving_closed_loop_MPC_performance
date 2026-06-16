**Tags:** #dataclass

## Overview

`MPCSolver` is a `NamedTuple` representing the fully compiled, ready-to-execute controller. It is the final artifact produced by the framework and what the user interacts with inside their high-speed simulation loops.

## Purpose

It exposes a clean, partitioned API for solving the optimal control problem in real-time. Instead of passing one massive dictionary of parameters to the solver, the user uses this class to independently update the slow-changing parameters (like a reference trajectory or mass matrix) and the fast-changing parameters (like the current state).

## Key Attributes & Methods

- **`prepare(slow_v)`**: Evaluates only the slow parametric terms and bakes them into the constant base QP. Returns a partially updated [[QPData]] payload.
- **`solve(fast_v, warmstart)`**: Evaluates only the fast parametric terms, patches them onto the base QP, and runs the numerical solver.
- **`solve_with_prepared(prepared_qp, fast_v, warmstart)`**: The highest-performance pathway. Takes the pre-computed QP from `prepare`, patches only the fast terms onto it, and solves.
- **`n_var` / `n_dec` / `n_eq` / `n_ineq`**: The finalized dimensions of the underlying optimization problem.

## Dependencies

- [[QPData]]: Used as the input/output payload for the matrix updates.
- [[Variable]]: Used for the internal dependency tracking map.