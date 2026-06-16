"""Closed-loop simulator for MPC (or any other controller).

A simulation is structurally divided into three distinct phases:

* **Init** — Executes once in Python before the loop. It processes the initial 
                 user inputs and returns the initial state (the "carry") for the loop.
* **Step** — Executes at every time step. Must be functionally pure as it is 
                 compiled into a ``jax.lax.scan``. It returns the updated carry 
                 and the data to be logged for that step.
* **Finalize** — Executes once in Python after the loop. It processes the final 
                 carry and the accumulated logs, returning the final simulation result.

The ``step`` function receives either the current step index ``k`` or a slice of a 
user-provided ``timeline`` PyTree. This enables time-varying dynamics, dynamic 
reference trajectories, or exogenous signals.

The simulator is strictly uncoupled from :class:`MPCProblem`; it works for any 
controller architecture. The user's ``step`` function typically closes over the 
MPC instance or dynamics models and calls them as needed.

Example
-------
::

    def init(inputs, n_steps):
        # Prepare the MPC problem once before the loop begins
        prepared = mpc.prepare({"Q_diag": inputs["Q_diag"]})
        return {"x": inputs["x0"], "prepared": prepared, "warmstart": None}

    def step(carry, k):
        # Solve the MPC step using the prepared QP
        sol  = mpc.solve({"x0": carry["x"]},
                         warmstart=carry["warmstart"],
                         prepared_qp=carry["prepared"])
        u0   = sol["x"]["u0"]
        
        # Advance the dynamics
        x_next = A_d @ carry["x"] + B_d @ u0
        
        new_carry = {**carry, "x": x_next, "warmstart": sol["status"]}
        log = {"x": carry["x"], "u": u0}
        
        return new_carry, log

    sim = ClosedLoop(init, step, n_steps=50)
    results = sim.run({"x0": jnp.array([1.0, 0.0]), "Q_diag": jnp.ones(2)})
"""

from __future__ import annotations
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp


class ClosedLoop:
    """A flexible, JAX-compiled simulation loop for dynamical systems.

    This class encapsulates a `jax.lax.scan` loop, providing hooks to 
    initialize state, execute control steps, and finalize data, while 
    automatically handling the temporal stacking of log variables.

    Parameters
    ----------
    init : Callable[[Any, int], Any]
        Function to initialize the simulation loop.
        Signature: ``init(inputs, n_steps) -> carry0``
    step : Callable[[Any, Any], tuple[Any, Any]]
        Function executed at each time step. Must be functionally pure.
        Signature: ``step(carry, x_user) -> (new_carry, log)``. 
        ``x_user`` is either the integer time step ``k``, or a time-slice 
        from the ``timeline`` provided to ``run``.
    n_steps : int
        The total number of time steps to simulate.
    finalize : Callable[[Any, Any], Any], optional
        Function to process the simulation results after the loop finishes.
        Signature: ``finalize(final_carry, stacked_logs) -> result``.
        If ``None``, the ``run`` method simply returns ``stacked_logs``.
    on_step : Callable[[int, Any], None], optional
        A Python callback invoked at every step via ``jax.debug.callback``.
        Useful for printing debug information or progress bars during the scan.
        Signature: ``on_step(k, log) -> None``.
    """

    def __init__(
        self,
        init:     Callable[[Any, int], Any],
        step:     Callable[[Any, Any], tuple[Any, Any]],
        n_steps:  int,
        finalize: Optional[Callable[[Any, Any], Any]] = None,
        on_step:  Optional[Callable[[int, Any], None]] = None,
    ) -> None:
        self._init     = init
        self._step     = step
        self._finalize = finalize
        self._n_steps  = n_steps
        self._on_step  = on_step

    def run(self, inputs: Any, timeline: Optional[Any] = None) -> Any:
        """Executes the closed-loop simulation.

        Parameters
        ----------
        inputs : Any
            The initial input data required by the ``init`` function to 
            construct the initial carry state.
        timeline : Any, optional
            A PyTree of arrays representing time-varying parameters (e.g., 
            reference trajectories, disturbances). Every leaf in this PyTree 
            must have a leading dimension equal to ``n_steps``. If provided, 
            the ``step`` function receives the slice ``timeline[k]`` at each 
            step. If ``None``, the ``step`` function receives the integer index ``k``.

        Returns
        -------
        Any
            The output of the ``finalize`` function, if provided. Otherwise, 
            returns the accumulated PyTree of ``logs``, where every leaf 
            has a new leading dimension of size ``n_steps``.
        """
        ks = jnp.arange(self._n_steps)

        if timeline is not None:
            _validate_timeline(timeline, self._n_steps)
            xs = (ks, timeline)           # scan body receives (k, slice)
            user_arg = lambda pair: pair[1]   # pass the slice to step
        else:
            xs = (ks, None)
            user_arg = lambda pair: pair[0]   # pass k to step

        def body(carry, pair):
            k      = pair[0]
            x_user = user_arg(pair)
            new_carry, log = self._step(carry, x_user)
            if self._on_step is not None:
                jax.debug.callback(self._on_step, k, log)
            return new_carry, log

        carry0 = self._init(inputs, self._n_steps)
        final_carry, logs = jax.lax.scan(body, carry0, xs)

        if self._finalize is None:
            return logs
        return self._finalize(final_carry, logs)


# ======================================================================
# Helpers
# ======================================================================

def _validate_timeline(timeline: Any, n_steps: int) -> None:
    """Validates that every leaf in a timeline PyTree spans the correct horizon.

    Parameters
    ----------
    timeline : Any
        The user-provided PyTree of time-varying arrays.
    n_steps : int
        The expected length of the leading dimension.

    Raises
    ------
    ValueError
        If the timeline is empty, or if any leaf does not have ``n_steps`` 
        as its leading dimension.
    """
    leaves = jax.tree_util.tree_leaves(timeline)
    if not leaves:
        raise ValueError("timeline pytree is empty.")
    for leaf in leaves:
        shape = jnp.shape(leaf)
        if not shape or shape[0] != n_steps:
            raise ValueError(
                f"Timeline leaves must have leading dimension {n_steps}, "
                f"got shape {shape}."
            )