**Tags:** #dataclass

## Overview

The `SlackSpec` class is an immutable data structure (dataclass) that defines how a hard constraint is relaxed into a soft constraint. It acts as a configuration payload that tells the MPC builder exactly how to append non-negative slack variables to the optimization problem.

## Purpose

In real-world control, unexpected disturbances or aggressive reference trajectories can make hard constraints (like state boundaries) physically impossible to satisfy, causing the QP solver to fail. `SlackSpec` solves this by allowing constraint violations at a steep mathematical penalty:

$$penalty_i = \frac{1}{2} w_{quad,i} \cdot s_i^2 + w_{lin,i} \cdot s_i$$

where $s_i \ge 0$ is the slack variable applied to a specific constraint row.

## Key Attributes

- **`rows`** (`Tuple[int, ...]`): A binary mask of length `n_cst` (total constraint rows). A `1` means the row gets a slack variable; a `0` means it remains a strict hard constraint.
- **`w_quad` & `w_lin`** (`Tuple[float, ...]`): The quadratic and linear penalty weights. Their lengths strictly match the number of `1`s in the `rows` mask (`n_slack`).    
- **`n_slack`** (`int`): The exact number of new decision variables this specification will add to the optimization problem.


## Dependencies

_This class is a foundational data structure. It acts as a pure leaf node and has no internal dependencies on other core framework classes._