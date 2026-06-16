**Tags:** #file

## Overview

The `_costs.py` module provides utility functions for constructing standard quadratic tracking costs. It easily handles penalizing deviations from reference states, controls, or outputs over the entire prediction horizon.

## Purpose

Similar to the bounds module, it separates the raw matrix math from the user-facing symbolic API:

1. **Builders** (`build_state_tracking`, `build_output_tracking`): Pure functions that construct the massive block-diagonal $P$ matrices, the linear $q$ vectors, and the constant $c$ offsets for tracking costs.
2. **Factories** (`state_tracking_cost`, `output_tracking_cost`): High-level wrappers that accept `Variable` inputs. They automatically handle time-tiling (broadcasting a single state penalty across the entire horizon) and return a ready-to-use [[Cost]] object.

## Dependencies

- [[Variable]]: Used to accept parametric reference trajectories or cost weight matrices.
- [[Cost]]: The factory outputs this object.