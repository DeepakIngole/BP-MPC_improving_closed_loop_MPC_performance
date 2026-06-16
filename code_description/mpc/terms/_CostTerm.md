**Tags:** #class #internal

## Overview

`_CostTerm` is an internal `NamedTuple` representing a single atomic callable contribution to the QP cost.

## Purpose

It decomposes a user-facing cost descriptor into its foundational mathematical pieces. Instead of treating a cost as a single block, this structure isolates whether a callable is updating the quadratic matrix (`P`), the linear vector (`q`), or the scalar offset (`c`). This isolation allows the assembler to track exactly which specific matrices need to be re-evaluated when a variable changes, avoiding unnecessary recomputations.

## Key Attributes

- **`target`**: Identifies which part of the cost this term computes (`"P"`, `"q"`, or `"c"`).
- **`fn`**: The mathematical callable.
- **`v_in`**: The specific variables required by this callable.
- **`uid`**: A unique identifier auto-generated for tracking in the computational graph.

## Dependencies

- [[Variable]]: Used to define the `v_in` dependency mapping.