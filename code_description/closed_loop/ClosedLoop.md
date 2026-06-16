**Tags:** #class

## Overview

The `closed_loop` module provides a generalized, JAX-accelerated simulation harness. While primarily designed to evaluate MPC controllers, its architecture is completely controller-agnostic. It wraps the highly efficient `jax.lax.scan` operation to run fast, compiled simulations over a given time horizon.

## Purpose

Writing custom `jax.lax.scan` loops for every new problem can be error-prone and tedious. The `ClosedLoop` class abstracts away this boilerplate by enforcing a strict three-phase structure:

1. **Init:** A pure Python setup phase that runs _before_ compilation (e.g., calling the MPC `prepare` method and defining the initial `carry` state).
    
2. **Step:** The functionally pure inner loop that executes the controller, steps the dynamics, and logs the current state.
    
3. **Finalize:** A pure Python teardown phase to unpack the JAX buffers, append the final state, and return a clean trajectory dictionary.
    

## Key Methods

- **`run(inputs, timeline)`**: Triggers the simulation. It seamlessly handles both static horizons and time-varying exogenous inputs (by slicing a provided `timeline` PyTree at each step and passing it to the user-defined `step` function).
    

## Dependencies

_This class is strictly uncoupled from the rest of the framework. It acts as a pure utility leaf node and has no internal dependencies on other core classes._