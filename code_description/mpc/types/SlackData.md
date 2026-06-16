**Tags:** #dataclass #internal

## Overview

`SlackData` is an internal `NamedTuple` that holds the precomputed constant arrays required for slack variable expansion.

## Purpose

When soft constraints are added to a problem, the QP matrices must be expanded to accommodate the new slack variables. Because the penalty weights are constant, this expansion data only needs to be computed exactly once during the `MPCProblem` initialization. `SlackData` stores these sub-matrices so the assembler can quickly concatenate them onto the main QP matrices at solve-time without recalculating them.

## Key Attributes

- **`n_slack`**: The total number of slack variables in the problem.
- **`P_slack`**: The diagonal quadratic penalty matrix for the slack variables.
- **`q_slack`**: The linear penalty vector for the slack variables.
- **`G_slack_cols`**: The specific columns to append to the inequality matrix `G` to enforce $s \ge 0$ and map the slacks to their respective constraint rows.

## Dependencies

_This class is a core data structure. It acts as a pure leaf node and has no internal dependencies on other framework classes._