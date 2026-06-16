**Tags:** #file

## Overview

The `problem.py` module is the architectural center of the framework. It defines the factorization logic that bridges the gap between the user's symbolic descriptors and the bare-metal array assemblers.

## Purpose

This file contains the strict logic required to extract atomic terms from lists of costs and constraints, categorize them by variable dependency, calculate the exact dimensions required for the solver (including slack variable expansions), and wire up the `jax.jit`-compatible solver functions.

## Dependencies

- It is the sole definition location for [[MPCProblem]] and [[MPCSolver]].
- Depends heavily on utility files like [[terms]] for term decomposition and extraction.
- Imports and routes to the backend assemblers ([[DenseAssembler]], [[SparseAssembler]]).