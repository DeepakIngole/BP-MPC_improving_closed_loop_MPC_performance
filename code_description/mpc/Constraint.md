**Tags:** #class

## Overview

The `Constraint` class is a core descriptor that represents a parametric linear constraint in the mathematical form:

$$LHS(v) \cdot z \begin{cases} == \\ \le \end{cases} RHS(v)$$

where $z$ is the decision vector, and $v$ represents a dictionary of runtime parameters.

## Purpose

This class acts as a container for the constraint logic. It separates the evaluation of the left-hand side matrix and the right-hand side vector, allowing the framework to determine which parts of the constraint are constant (compiled once) and which parts are parametric (re-evaluated during the control loop). It also handles the mathematical stacking of constraints and the conversion of hard constraints into soft constraints via slacking.

## Key Features & Methods

- **`eval_lhs(v)` & `eval_rhs(v)`**: Safely executes the internal lambdas to generate the concrete JAX matrices and vectors using the provided dictionary of runtime parameters.
- **`add_slack(rows, w_quad, w_lin)`**: A fluent modifier method. It returns a new, modified copy of the inequality constraint that includes a `SlackSpec`, seamlessly converting it from a hard constraint into a soft constraint penalized in the cost function.
- **Constraint Stacking (`__add__`)**: Overloads the `+` operator, allowing users to vertically concatenate multiple `Constraint` objects into a single, unified constraint block.

## Dependencies

- [[Variable]]: Used to specify which runtime variables the $LHS$ and $RHS$ functions depend on (`v_in_lhs`, `v_in_rhs`).   
- [[SlackSpec]]: Used internally when the `.slack()` method is called to define the weights and indexing for soft constraint violations.
