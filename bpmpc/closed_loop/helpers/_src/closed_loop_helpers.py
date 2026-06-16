from bpmpc_jax.closed_loop import ClosedLoop, RunLogger
from typing import Optional, Any, Callable, Iterable, List

import jax
import jax.numpy as jnp

import time
import jax
import jax.numpy as jnp
import optax
from typing import Callable, Any, Optional

import time
from typing import Any, Callable, Iterable, Optional
import jax
import jax.tree_util as jtu
import optax

def closed_loop_tune(
    loss_fn: Callable,
    initial_params: Any,
    optimizer: Optional[optax.GradientTransformation] = None,
    n_iters: int = 50,
    dataloader: Optional[Iterable[Any]] = None,
    batch_size: int = 1,
    has_aux: bool = False,
):
    """
    Optimizes arbitrary parameters using a differentiable loss function.
    
    Parameters
    ----------
    loss_fn : Callable
        If `dataloader` is None, expected signature is `loss_fn(params)`.
        If `dataloader` is provided, expected signature is `loss_fn(params, item)`.
        If `has_aux=True`, the function must return `(loss, aux_data)`.
    initial_params : Any
        The starting parameters (can be a dict, dataclass, or array).
    optimizer : optax.GradientTransformation, optional
        The Optax optimizer to use. Defaults to Adam with gradient clipping.
    n_iters : int
        Number of optimization steps (parameter updates).
    dataloader : Iterable[Any], optional
        An iterator or generator yielding individual data items.
    batch_size : int, optional
        Number of individual items to accumulate gradients over before making 
        an optimizer update.
    has_aux : bool, optional
        If True, expects `loss_fn` to return a tuple `(loss, aux_data)`. 
        Collects `aux_data` over all iterations.
        
    Returns
    -------
    params : Any
        The optimized parameters.
    aux_history : List or None
        The collected auxiliary data if `has_aux=True`, otherwise `None`.
    """
    if optimizer is None:
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(learning_rate=1e-2)
        )

    opt_state = optimizer.init(initial_params)
    logger = RunLogger()
    params = initial_params
    
    # Initialize as a list only if we are tracking aux data
    aux_history = []

    # ==========================================
    # BRANCH 1: DETERMINISTIC MODE (No Batching)
    # ==========================================
    if dataloader is None:
        backward = jax.jit(jax.value_and_grad(loss_fn, has_aux=has_aux))
        
        def train_step_single(p, state):
            if has_aux:
                (loss_val, aux), grads = backward(p)
            else:
                loss_val, grads = backward(p)
                aux = None
                
            updates, new_state = optimizer.update(grads, state, p)
            new_p = optax.apply_updates(p, updates)
            return new_p, new_state, loss_val, aux

        print("Starting differentiable tuning (Deterministic mode)...")
        for epoch in range(n_iters):
            t0 = time.time()
            params, opt_state, loss_val, aux = train_step_single(params, opt_state)
            
            if has_aux:
                aux_history.append(aux)
                
            dt = time.time() - t0
            logger.log(Epoch=f"{epoch:02d}", TaskCost=f"{loss_val:.2f}", Time=f"{dt:.3f}s")

    # ==========================================
    # BRANCH 2: STOCHASTIC MODE (Gradient Accumulation)
    # ==========================================
    else:
        # argnums=0 ensures gradients are only taken w.r.t. parameters
        backward = jax.jit(jax.value_and_grad(loss_fn, argnums=0, has_aux=has_aux))
        
        print(f"Starting differentiable tuning (Gradient Accumulation, batch_size={batch_size})...")
        batch_iter = iter(dataloader)
        
        # n_iters now represents the number of actual parameter updates
        for step in range(n_iters):
            t0 = time.time()
            
            accumulated_grads = None
            total_loss = 0.0
            step_aux = []
            
            # 1. Accumulate gradients over the specified batch_size
            for _ in range(batch_size):
                try:
                    item = next(batch_iter)
                except StopIteration:
                    batch_iter = iter(dataloader)
                    item = next(batch_iter)
                
                if has_aux:
                    (loss_val, aux), grads = backward(params, item)
                    step_aux.append(aux)
                else:
                    loss_val, grads = backward(params, item)
                    
                total_loss += loss_val
                
                # JAX Pytree accumulation: add current grads to the running total
                if accumulated_grads is None:
                    accumulated_grads = grads
                else:
                    accumulated_grads = jtu.tree_map(lambda x, y: x + y, accumulated_grads, grads)
            
            if has_aux:
                aux_history.append(step_aux)
            
            # 2. Average the loss and gradients
            avg_loss = total_loss / batch_size
            avg_grads = jtu.tree_map(lambda x: x / batch_size, accumulated_grads)
            
            # 3. Apply the single batched update
            updates, opt_state = optimizer.update(avg_grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            
            dt = time.time() - t0
            logger.log(Step=f"{step:02d}", TaskCost=f"{avg_loss:.2f}", Time=f"{dt:.3f}s")

    print("\nTuning complete.")
    
    # Always return a 2-tuple
    return params, aux_history


def build_closed_loop_simulator(
    mpc, 
    plant, 
    trajectory_cost_fn, 
    horizon_sim: int, 
    horizon_mpc: int, 
    nx: int, 
    nu: int,
    runtime_keys: Optional[List[str]] = None,
    options: Optional[dict] = None
) -> ClosedLoop:
    """
    Constructs a ClosedLoop simulation object for an MPC-controlled system.

    Parameters
    ----------
    mpc : object
        The Model Predictive Control object.
    plant : object
        The true system dynamics object.
    trajectory_cost_fn : callable
        Function returning a scalar cost from `(xs, us)`.
    horizon_sim : int
        The total number of timesteps to simulate.
    horizon_mpc : int
        The prediction horizon length used by the MPC.
    nx : int
        The dimension of the state vector.
    nu : int
        The dimension of the control input vector.
    runtime_keys: list of str, optional
        A list of parameter names in `inputs` that should be treated as 
        time-varying sequences of length `horizon_sim`. These will be 
        sliced at each timestep `k`.
    options : dict, optional
        Additional configuration options.
    """
    if options is None:
        options = {}
        
    warmstart_first_mpc = options.get("warmstart_first_mpc", False)
    lin_mode = options.get("linearization_mode", "none")

    valid_modes = ["trajectory_shifted", "trajectory", "current_state", "none"]
    if lin_mode not in valid_modes:
        raise ValueError(f"Unknown linearization_mode: '{lin_mode}'. Valid options: {valid_modes}")

    mpc_keys = set(mpc.all_vars.keys())
    plant_keys = set(plant.true_params_spec.keys())
    rt_keys_set = set(runtime_keys) if runtime_keys is not None else set()

    def init(inputs, n_steps):
        x0 = inputs["x0"]
        
        # --- Separate static vs runtime parameters for MPC ---
        mpc_static = {k: inputs[k] for k in mpc_keys if k in inputs and k not in rt_keys_set}
        mpc_runtime = {k: inputs[k] for k in mpc_keys if k in inputs and k in rt_keys_set}
        
        # --- Separate static vs runtime parameters for Plant ---
        plant_static = {k: inputs[k] for k in plant_keys if k in inputs and k not in rt_keys_set}
        plant_runtime = {k: inputs[k] for k in plant_keys if k in inputs and k in rt_keys_set}

        # Evaluate runtime sequences at k=0 for initial preparation
        mpc_params_k0 = {**mpc_static}
        for key, seq in mpc_runtime.items():
            mpc_params_k0[key] = seq[0]

        prep = mpc.prepare(mpc_params_k0)
        solve_args = {"x0": x0, **mpc_params_k0}
        
        carry = {
            "x": x0, 
            "prepared": prep,
            "mpc_static": mpc_static,
            "mpc_runtime": mpc_runtime,
            "plant_static": plant_static,
            "plant_runtime": plant_runtime
        }
        
        if lin_mode != "none":
            x_seed = jnp.tile(x0, (horizon_mpc, 1))
            u_seed = jnp.zeros((horizon_mpc, nu))
            solve_args["x_nom"] = x_seed
            solve_args["u_nom"] = u_seed
        
        if warmstart_first_mpc:
            sol0 = mpc.solve_with_prepared(prep, solve_args, warmstart=None)
            if lin_mode != "none":
                carry["x_nom"] = sol0["x"]["x"].reshape((horizon_mpc, nx))
                carry["u_nom"] = sol0["x"]["u"].reshape((horizon_mpc, nu))
        else:
            if lin_mode != "none":
                carry["x_nom"] = x_seed
                carry["u_nom"] = u_seed
            
        return carry

    def step(carry, k):
        # --- Assemble current MPC parameters at step k ---
        current_mpc_params = {**carry["mpc_static"]}
        for key, seq in carry["mpc_runtime"].items():
            current_mpc_params[key] = seq[k]
            
        solve_args = {"x0": carry["x"], **current_mpc_params}
        
        if lin_mode != "none":
            solve_args["x_nom"] = carry["x_nom"]
            solve_args["u_nom"] = carry["u_nom"]

        sol = mpc.solve_with_prepared(carry["prepared"], solve_args, warmstart=None)
        u = sol["x"]["u0"]
        
        # --- Assemble current Plant parameters at step k ---
        current_plant_params = {**carry["plant_static"]}
        for key, seq in carry["plant_runtime"].items():
            current_plant_params[key] = seq[k]
        
        x_next = plant.step(carry["x"], u, true_params=current_plant_params)
        
        new_carry = {
            "x": x_next, 
            "prepared": carry["prepared"],
            "mpc_static": carry["mpc_static"],
            "mpc_runtime": carry["mpc_runtime"],
            "plant_static": carry["plant_static"],
            "plant_runtime": carry["plant_runtime"]
        }
        
        if lin_mode != "none":
            x_sol = sol["x"]["x"].reshape((horizon_mpc, nx))
            u_sol = sol["x"]["u"].reshape((horizon_mpc, nu))
            
            if lin_mode == "trajectory_shifted":
                next_u_nom = jnp.concatenate([u_sol[1:], u_sol[-1:]], axis=0)
                next_x_nom = x_sol.at[0].set(x_next)
            elif lin_mode == "trajectory":
                next_u_nom = u_sol
                next_x_nom = x_sol.at[0].set(x_next)
            elif lin_mode == "current_state":
                u_next_scalar = u_sol[1] if horizon_mpc > 1 else u_sol[0]
                next_u_nom = jnp.tile(u_next_scalar, (horizon_mpc, 1))
                next_x_nom = jnp.tile(x_next, (horizon_mpc, 1))
            
            new_carry["x_nom"] = next_x_nom
            new_carry["u_nom"] = next_u_nom

        return new_carry, {"x": carry["x"], "u": u}

    def finalize(final_carry, logs):
        xs, us = logs["x"], logs["u"]
        cost = trajectory_cost_fn(xs, us)
        return {"cost": cost, "x": xs, "u": us}

    return ClosedLoop(init=init, step=step, n_steps=horizon_sim, finalize=finalize)