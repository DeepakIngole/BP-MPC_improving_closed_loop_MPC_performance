**Tags:** #file

## Overview

The `_bounds.py` module provides utility functions for constructing state and control input box constraints for the MPC problem. It abstracts away the tedious matrix formatting required to bound the decision vector.

## Purpose

It operates in two distinct layers:

1. **Builders** (`build_box_lhs`, `build_box_rhs`): Pure JAX functions that take concrete numerical arrays (min/max bounds) and return the fully constructed dense matrices required by the QP solver.
2. **Factories** (`box_bounds`): A high-level wrapper that accepts symbolic descriptors (like `Variable` or concrete arrays). It automatically resolves parameters at runtime, handles default infinite bounds (`jnp.inf`), sets up slack variables if requested, and returns a fully initialized **[[Constraint]]** object.

## Dependencies

- [[Variable]]: Used to accept parametric bound limits.   
- [[Constraint]]: The factory outputs this object.
- [[SlackSpec]]: Used internally to relax the box constraints into soft constraints if the user provides slack weights.