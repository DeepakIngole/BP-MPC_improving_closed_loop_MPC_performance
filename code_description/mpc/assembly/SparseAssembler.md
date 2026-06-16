**Tags:** #class #assembler

## Overview

The `SparseAssembler` class is an advanced backend engine that builds and updates the optimization problem using XLA-fused `jax.experimental.sparse.BCOO` matrices.

## Purpose

For problems with long horizons or large state spaces, dense matrices are mostly filled with zeros, wasting memory and computation. This assembler statically discovers the exact structural sparsity pattern of the MPC problem at build-time. It does this via **random probing**—feeding dummy random noise through the user's cost and constraint functions to see exactly which matrix coordinates light up as non-zero.

Once the pattern is locked in, the `apply` method is capable of performing lightning-fast, XLA-optimized updates. It strictly updates the 1D arrays of non-zero data without ever having to rebuild the underlying graph connectivity.

## Key Features & Methods

- **Random Probing (`_probe_sparsity`)**: Automatically discovers the structural footprint of the problem so the user doesn't have to manually track matrix non-zeros.
- **`matrix_metadata`**: Stores the static `BCOO` indices and exact memory slices where each atomic term will write its data.
- **`apply(v, fast_v)`**: Rapidly evaluates parametric terms and routes their outputs directly into the flat, packed 1D data arrays of the `BCOO` format.

## Dependencies

- [[QPData]]: Used as the primary container, though here its matrix fields (`P`, `A`, `G`) output as `BCOO` objects instead of dense arrays.
- [[_CostTerm]] & [[_CstTerm]]: Evaluated and mapped to the sparse memory footprint.
- [[Constraint]]: Scanned for structural slack variable allocations.