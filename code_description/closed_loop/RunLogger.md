**Tags:** #class

## Overview

The `RunLogger` class is a dynamic terminal logging utility designed to format runtime metrics into a clean, dynamically sized ASCII table.

## Purpose

During iterative processes like control loops, optimization algorithms, or hyperparameter tuning, it is critical to track ongoing metrics. `RunLogger` abstracts away the formatting boilerplate. It calculates column widths automatically based on the length of the metric names, formats floating-point numbers into clean scientific notation, and smartly reprints the table header only if the set of tracked fields changes mid-run.

## Key Methods

- **`log(**metrics)`**: The primary user-facing method. Takes arbitrary keyword arguments (e.g., `log(iter=1, cost=0.5)`) and prints them as a formatted row.
- **`_update_layout(metrics)`**: Recalculates the required width for each column to ensure perfect alignment.
- **`_print_header()` & `_print_row(metrics)`**: Internal helpers that handle the actual string formatting and terminal output.

## Dependencies

_This class is a pure utility leaf node and has no internal dependencies on other core framework classes. It relies only on standard Python libraries and `numpy`._