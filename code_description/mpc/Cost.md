**Tags:** #class

## Overview

The `Cost` class is a core descriptor that represents a parametric quadratic cost function in the standard mathematical form:

$$Cost(v, z) = \frac{1}{2} z^T Q(v) z + q(v)^T z + c(v)$$

where $z$ is the decision vector, and $v$ represents a dictionary of runtime parameters.

## Purpose

Like the constraint descriptor, this class separates the evaluation of the quadratic ($Q$), linear ($q$), and constant ($c$) terms. This granular separation is critical for performance: it allows the framework to know exactly which matrices need to be rebuilt when a specific parameter changes, and which can be compiled exactly once. If a term is mathematically zero (e.g., no linear cost), the class automatically handles it as a zero array to keep the API uniform for the solver.

## Key Features & Methods

- **`eval_q_mat(v)`, `eval_q_vec(v)`, `eval_c(v)`**: Safely executes the internal lambdas to generate the concrete JAX arrays for each specific term using the provided dictionary of runtime parameters.
- **Cost Stacking (`__add__`)**: Overloads the `+` operator. This allows users to modularly define separate costs (e.g., a state tracking cost + a control effort cost) and simply add them together `c_total = c1 + c2`. The framework automatically merges the math and the dependency dictionaries under the hood.

## Dependencies

- [[Variable]]: Used to specify exactly which runtime variables each specific mathematical term depends on (`v_in_q_mat`, `v_in_q_vec`, `v_in_c`).