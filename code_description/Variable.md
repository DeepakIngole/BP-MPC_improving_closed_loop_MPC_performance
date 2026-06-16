**Tags:** #class

## Overview

The `Variable` class is a foundational symbolic descriptor in the MPC framework. It is a lightweight placeholder that carries **no numerical data**. Instead, it defines the expected structure of parameters that will be provided at runtime.

## Purpose

It tells the MPC builder which arrays the user's constraint and cost functions depend on. By using `Variable` objects during the problem definition phase, the framework can trace dependencies, validate array shapes, and automatically wire up the JAX computational graph before any actual data is passed into the solver.

## Key Attributes

- **`name`** (`str`): A human-readable identifier. This acts as the unique dictionary key used to pass actual JAX arrays at runtime.
- **`shape`** (`Tuple[int, ...]`): The exact expected dimensions of the concrete JAX array this variable represents (e.g., `(1,)` for scalars, `(N, nx)` for trajectories).

## Dependencies

_This class is a core leaf node and has no internal dependencies on other framework classes._