**Tags:** #dataclass #internal

## Overview

`QPData` is an internal `NamedTuple` that serves as the standard container for the seven numeric arrays defining a convex Quadratic Program (QP).

## Purpose

It provides a unified, structured payload to pass the fully assembled optimization problem from the MPC builder directly to the backend mathematical solver. It is designed to be mode-agnostic, meaning its matrix fields (`P`, `A`, `G`) can hold either dense `jax.Array` objects or sparse `BCOO` objects depending on how the problem was compiled, while keeping the API consistent.

## Key Attributes

- **`P`, `q`, `c`**: The quadratic matrix, linear vector, and scalar offset defining the cost function.
- **`A`, `b`**: The matrix and vector defining the equality constraints ($A z = b$).
- **`G`, `h`**: The matrix and vector defining the inequality constraints ($G z \le h$).

## Dependencies

_This class is a core data structure. It acts as a pure leaf node and has no internal dependencies on other framework classes._