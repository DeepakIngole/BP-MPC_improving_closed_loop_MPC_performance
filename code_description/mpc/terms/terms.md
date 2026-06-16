**Tags:** #file

## Overview

The `terms.py` module handles the translation of high-level user constructs into the atomic operations required by the solver. It is the sole definition location for the **[[_CostTerm]]** and **[[_CstTerm]]** internal data structures.

## Purpose

This file provides the necessary utility functions (`decompose_costs` and `decompose_constraints`) to flatten a user's lists of macro costs and constraints into a massive, unordered list of independent, atomic terms. It also provides `collect_variables` to scan all of these terms and gather every unique variable dependency, ensuring there are no naming collisions before the assembly phase begins.

## Dependencies

- [[Cost]]: Decomposed by this file.
- [[Constraint]]: Decomposed by this file.
- [[Variable]]: Gathered and validated by this file.