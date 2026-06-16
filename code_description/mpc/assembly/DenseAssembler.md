**Tags:** #class

## Overview

The `DenseAssembler` class is the backend engine for building and updating the optimization problem using standard, contiguous `jax.numpy` arrays.

## Purpose

It avoids the overhead of allocating massive new matrices at every time step of a control loop. During the MPC problem's initial build phase, it pre-allocates a `base_qp` that is perfectly sized to the fully expanded decision dimension (including all slack variables). It mathematically bakes all the constant problem parameters and slack weights directly into this base layer.

At runtime, the `apply` method acts as an incredibly fast differential patcher. It evaluates only the changing parametric terms and functionally accumulates them on top of the `base_qp` using JAX's non-destructive scatter operations (`.at[...].add(...)`).

## Key Attributes & Methods

- **`base_qp`**: The pre-sized, pre-computed foundation of the problem holding all constant data and slack matrix expansions.
- **`apply(v, fast_v)`**: The runtime workhorse. It takes the dictionaries of slow and fast variables, evaluates their specific mathematical terms, and scatters the results into a fresh copy of the QP matrices.

## Dependencies

- [[QPData]]: Used as the primary data container for both `base_qp` and the returned runtime matrices.
- [[_CostTerm]] & [[_CstTerm]] (via `Term`): The atomic operations that the assembler executes.
- [[Constraint]]: Scanned during build-time to extract and bake in slack variables. 
- [[Variable]]: Used to resolve parameter dependencies. 