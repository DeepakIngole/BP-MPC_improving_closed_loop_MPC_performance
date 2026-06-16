**Tags:** #class #internal

## Overview

`_CstTerm` is an internal `NamedTuple` representing a single atomic callable contribution to a QP constraint block.

## Purpose

Similar to `_CostTerm`, it breaks down a macro constraint descriptor into its left-hand side (`lhs`) or right-hand side (`rhs`) components. Crucially, it also carries strict indexing data so the assembler knows exactly where to write this term's output within the massive, flattened QP matrices.

## Key Attributes

- **`target`**: Identifies if this term computes a left-hand side matrix (`"lhs"`) or right-hand side vector (`"rhs"`).
- **`kind`**: Indicates if it writes to the equality (`"eq"`) or inequality (`"ineq"`) constraint blocks.
- **`rs` / `re`**: The start and end row indices defining exactly where this term belongs in the global QP problem.

## Dependencies

- [[Variable]]: Used to define the `v_in` dependency mapping.