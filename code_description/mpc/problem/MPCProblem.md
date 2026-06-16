**Tags:** #class

## Overview

The `MPCProblem` class is the main analytical engine of the framework. It takes the user's high-level definitions of costs and constraints and factorizes them into an optimized internal representation, completely setting up the optimization problem before any actual control loop starts.

## Purpose

Its primary job is to maximize performance under JAX's JIT compilation. It does this by analyzing the dependencies of all mathematical terms and strictly partitioning them into three categories:

1. **Constant Terms**: Evaluated exactly once at build-time.
2. **Slow Terms**: Evaluated infrequently (e.g., once per episode) via the `prepare` step.
3. **Fast Terms**: Evaluated at every single control step via the `solve` step.

By doing this, it guarantees that the solver never recalculates a matrix unless its specific parameters have changed. It is also responsible for automatically inferring the global matrix shapes and expanding them to accommodate any defined soft constraints (slack variables).

## Key Methods

- **`with_solver(solver_fn)`**: Takes a backend numerical QP solver (like Jaxsparrow or OSQP) and wraps the assembled problem around it, returning a ready-to-use [[MPCSolver]].

## Dependencies

- [[Cost]]: Ingested and decomposed during problem creation.
- [[Constraint]]: Ingested and decomposed during problem creation.
- [[DenseAssembler]] / [[SparseAssembler]]: Instantiated internally to handle the actual matrix patching depending on the requested `mode`.