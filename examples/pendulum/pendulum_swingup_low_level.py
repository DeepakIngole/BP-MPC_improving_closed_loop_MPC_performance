"""
Differentiable MPC: tuning cost weights via gradient descent
============================================================

This example shows how to use ``bpmpc_jax`` to tune the *internal* cost
weights of a Model Predictive Controller end-to-end, so that the resulting
*closed-loop* behaviour is optimal with respect to a separate task cost.

Setup
-----
* **Plant**: cart-pole, starting at the hang-down equilibrium. Goal: swing up
  to upright.
* **Inner controller (MPC)**:
    - Horizon ``HORIZON_MPC`` steps with sampling time ``DT``.
    - Stage cost ``Q_RUN`` is fixed.
    - Terminal weight ``Q_term`` and input weight ``R`` are parameterised by
      ``theta`` and *learned*.
    - One SQP iteration per control step, warm-started from the previous solve.
* **Outer optimiser**: Optax SGD with a decaying learning rate. The loss is
  the LQR-style task cost plus a soft penalty on state-bound violations,
  evaluated over the full ``HORIZON_SIM`` rollout. Gradients flow through
  every QP solve in the rollout.

The script prints a per-epoch training log and ends by plotting cart position,
pole angle, and control input *before* and *after* optimisation.
"""

import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from scipy.linalg import solve_discrete_are

jax.config.update("jax_enable_x64", True)

from bpmpc_jax.variable import Variable
from bpmpc_jax.mpc import MPCProblem, Cost
from bpmpc_jax.mpc.helpers import build_state_tracking, nonlinear_dynamics, box_bounds
from bpmpc_jax.closed_loop import ClosedLoop, RunLogger
from bpmpc_jax.env import CartPendulum

from jaxsparrow import setup_sparse_solver

# ============================================================================
# 1. Hyperparameters
# ============================================================================
# --- Plant / closed-loop simulation ---
DT          = 0.015
HORIZON_SIM = 170                                          # rollout length

# --- MPC ---
HORIZON_MPC = 11
NX, NU      = 4, 1
TERM_REG    = 1e-2                                         # regularisation on Q_term
R_REG       = 1e-6                                         # regularisation on R

X_MIN_MPC = jnp.array([-5.0, -5.0, -jnp.inf, -jnp.inf])    # MPC state bounds
X_MAX_MPC = jnp.array([ 5.0,  5.0,  jnp.inf,  jnp.inf])
U_MIN_MPC = jnp.array([-4.0])                              # MPC input bounds
U_MAX_MPC = jnp.array([ 4.0])

Q_RUN = jnp.diag(jnp.array([100.0, 1.0, 100.0, 1.0]))      # fixed running cost

# --- Task cost (defines the outer/meta-loss) ---
X0_INIT = jnp.array([0.0, 0.0, -jnp.pi, 0.0])              # x, dx, theta, dtheta
Q_TASK  = jnp.diag(jnp.array([100.0, 1.0, 100.0, 1.0]))
R_TASK  = jnp.diag(jnp.array([1e-6]))
X_MIN_TASK  = jnp.array([-5.0, -5.0, -jnp.inf, -jnp.inf])
X_MAX_TASK  = jnp.array([ 5.0,  5.0,  jnp.inf,  jnp.inf])
VIOLATION_W = 100.0                                        # penalty weight

# --- Learnable parameter layout ---
# theta is split into (c_q, c_r):
#   c_q : lower-triangular Cholesky factor of (Q_term - TERM_REG * I)
#   c_r : per-input weight; R = diag(c_r**2) + R_REG * I
# Both encodings keep the matrices positive (semi-)definite for any value of
# theta, so no projection is needed during optimisation.
NX_TRI = NX * (NX + 1) // 2                                # 10
NP     = NX_TRI + NU                                       # 11

# --- Outer optimiser ---
N_ITER = 200
RHO    = 1e-4
ETA    = 0.51
CLIP   = 1e4


# ============================================================================
# 2. Building the MPC problem
# ============================================================================
def setup_swingup_mpc(plant):
    """Build the swing-up MPC and bind a sparse, differentiable QP solver."""

    # --- Variables ---
    # Slow variable: only changes when the outer optimiser updates theta.
    p_var = Variable("p", shape=(NP,))

    # Fast variables: refreshed at every closed-loop step.
    x0_var    = Variable("x0",    shape=(NX,))
    x_nom_var = Variable("x_nom", shape=(HORIZON_MPC, NX))
    u_nom_var = Variable("u_nom", shape=(HORIZON_MPC, NU))

    # --- Cost: pure quadratic in the decision variable ---
    n_z = HORIZON_MPC * NX + HORIZON_MPC * NU
    tril_idx = jnp.tril_indices(NX)

    def _terminal_from_cq(c_q):
        L = jnp.zeros((NX, NX)).at[tril_idx].set(c_q)
        return L @ L.T + TERM_REG * jnp.eye(NX)

    def _get_P(v):
        c_q = v["p"][:NX_TRI]
        c_r = v["p"][NX_TRI:]
        Q_term = _terminal_from_cq(c_q)
        R_mat  = jnp.diag(c_r**2 + R_REG)

        # Stage costs: Q_RUN at k = 0..H-1, Q_term at k = H.
        Q_tiled = jnp.broadcast_to(Q_RUN, (HORIZON_MPC + 1, NX, NX))
        Q_tiled = Q_tiled.at[-1].set(Q_term)
        R_tiled = jnp.broadcast_to(R_mat, (HORIZON_MPC, NU, NU))

        P, _, _ = build_state_tracking(
            Q_tiled, R_tiled,
            jnp.zeros((HORIZON_MPC + 1, NX)),
            jnp.zeros((HORIZON_MPC, NU)),
            jnp.zeros((NX,)),
            HORIZON_MPC,
        )
        return P

    cost = Cost(
        q_mat=_get_P,
        q_vec=jnp.zeros(n_z),
        c=jnp.array(0.0),
        v_in_q_mat={"p": p_var},
        v_in_q_vec=None,
        v_in_c=None,
    )

    # --- Constraints: nonlinear plant dynamics + box bounds ---
    dyn = nonlinear_dynamics(
        dyn=plant,
        x_nominal=x_nom_var,
        u_nominal=u_nom_var,
        horizon=HORIZON_MPC,
    )
    # add slack to state constraints only
    bnds_state = box_bounds(
        x_min=X_MIN_MPC, x_max=X_MAX_MPC,
        n_u = U_MIN_MPC.shape[0],
        horizon=HORIZON_MPC,
    ).add_slack(w_quad=10,w_lin=10)
    bnds_input = box_bounds(
        u_min=U_MIN_MPC, u_max=U_MAX_MPC,
        n_x = X_MIN_MPC.shape[0],
        horizon=HORIZON_MPC,
    )

    mpc = MPCProblem(
        costs=[cost],
        constraints=[dyn, bnds_state, bnds_input],
        outputs={
            "u0": slice(HORIZON_MPC * NX,       HORIZON_MPC * NX + NU),
            "x":  slice(0,                      HORIZON_MPC * NX),
            "u":  slice(HORIZON_MPC * NX,       HORIZON_MPC * (NX + NU)),
        },
        mode="sparse",
    )

    solver = setup_sparse_solver(
        n_var=mpc.n_dec,
        n_ineq=mpc.n_ineq,
        n_eq=mpc.n_eq,
        sparsity_patterns=mpc.sparsity_pattern,
        fixed_elements=mpc.fixed_elements,
        options={"diff_mode": "rev"},
    )
    return mpc.with_solver(solver)


# Build plant and MPC once at module level so they are shared by the loss and
# the standalone rollout used for plotting.
env   = CartPendulum(dt=DT)
plant = env.get_dynamics()
mpc   = setup_swingup_mpc(plant)


# ============================================================================
# 3. Closed-loop simulation
# ============================================================================
def simulate_closed_loop(p):
    """Run the closed-loop MPC simulation for a given parameter ``p``.

    Returns a dict with:
      * ``cost`` : scalar task cost (objective for the outer optimiser).
      * ``xs``   : (HORIZON_SIM, NX) state trajectory.
      * ``us``   : (HORIZON_SIM, NU) input trajectory.
    """

    def init(inputs, n_steps):
        prep = mpc.prepare({"p": p})

        # Crude initial guess for the very first SQP linearisation point.
        x_seed = jnp.tile(inputs["x0"], (HORIZON_MPC, 1))
        u_seed = jnp.zeros((HORIZON_MPC, NU))

        # One full MPC solve to obtain a useful linearisation trajectory
        # before the closed loop starts.
        sol0 = mpc.solve_with_prepared(
            prep,
            {"x0": inputs["x0"], "x_nom": x_seed, "u_nom": u_seed, "p": p},
            warmstart=None,
        )
        x_nom = sol0["x"]["x"].reshape((HORIZON_MPC, NX))
        u_nom = sol0["x"]["u"].reshape((HORIZON_MPC, NU))

        return {"x": inputs["x0"], "x_nom": x_nom, "u_nom": u_nom, "prepared": prep}

    def step(carry, k):
        # A. Solve the MPC (one SQP iteration).
        sol = mpc.solve_with_prepared(
            carry["prepared"],
            {"x0": carry["x"], "x_nom": carry["x_nom"],
             "u_nom": carry["u_nom"], "p": p},
            warmstart=None,
        )
        u = sol["x"]["u0"]

        # B. Step the true environment.
        x_next = plant.step(carry["x"], u)

        # C. Build the linearisation trajectory for the NEXT step.
        # ---- Original (kept for reference) ----
        # next_x_nom = sol["x"]["x"].reshape((HORIZON_MPC, NX))
        # next_u_nom = sol["x"]["u"].reshape((HORIZON_MPC, NU))
        x_sol = sol["x"]["x"].reshape((HORIZON_MPC, NX))
        u_sol = sol["x"]["u"].reshape((HORIZON_MPC, NU))
        next_u_nom = jnp.concatenate([u_sol[1:], u_sol[-1:]], axis=0)
        next_x_nom = x_sol.at[0].set(x_next)

        new_carry = {
            "x":        x_next,
            "x_nom":    next_x_nom,
            "u_nom":    next_u_nom,
            "prepared": carry["prepared"],
        }
        return new_carry, {"x": carry["x"], "u": u}

    def finalize(final_carry, logs):
        xs, us = logs["x"], logs["u"]

        # Quadratic task cost.
        track = (jnp.einsum("ti,ij,tj->", xs, Q_TASK, xs)
                 + jnp.einsum("ti,ij,tj->", us, R_TASK, us))

        # Soft L1 penalty on state-bound violation.
        upper = jnp.where(jnp.isfinite(X_MAX_TASK),
                          jnp.maximum(0.0, xs - X_MAX_TASK), 0.0)
        lower = jnp.where(jnp.isfinite(X_MIN_TASK),
                          jnp.maximum(0.0, X_MIN_TASK - xs), 0.0)
        viol  = jnp.sum(upper) + jnp.sum(lower)

        return {"cost": track + VIOLATION_W * viol, "xs": xs, "us": us}

    sim = ClosedLoop(init=init, step=step,
                     n_steps=HORIZON_SIM, finalize=finalize)
    return sim.run({"x0": X0_INIT})


def compute_trajectory_loss(theta_tune):
    """Scalar objective for the outer optimiser."""
    return simulate_closed_loop(theta_tune["p"])["cost"]


# ============================================================================
# 4. Initialisation: DARE for a sane starting theta
# ============================================================================
def dare_init_theta(plant):
    """Initialise theta from the discrete algebraic Riccati equation.

    Linearise the plant at the upright equilibrium, solve the DARE for the
    terminal cost, and Cholesky-factorise the result into the c_q encoding.
    """
    x0 = jnp.zeros(NX)
    u0 = jnp.zeros(NU)
    A = np.array(jax.jacfwd(plant.step, argnums=0)(x0, u0))
    B = np.array(jax.jacfwd(plant.step, argnums=1)(x0, u0))

    P_dare = solve_discrete_are(
        A, B,
        np.diag([100.0, 1.0, 100.0, 1.0]),
        np.array([[1e-6]]),
    )
    L   = np.linalg.cholesky(P_dare - TERM_REG * np.eye(NX))
    c_q = L[np.tril_indices(NX)]
    c_r = np.array([1e-3])
    return jnp.array(np.concatenate([c_q, c_r]))


# ============================================================================
# 5. Optax tuning loop
# ============================================================================
def tune_mpc_weights(theta_initial):
    """Tune MPC parameters by SGD on the closed-loop task cost.

    Parameters
    ----------
    theta_initial : dict
        Pytree of starting parameters, e.g. ``{"p": jnp.ndarray}``.

    Returns
    -------
    theta_final : dict
        Pytree of optimised parameters in the same structure as the input.
    """
    theta = theta_initial

    # Robbins-Monro-style decaying step size.
    def alpha_schedule(k):
        return RHO * jnp.log(k + 2) / (k + 1) ** ETA

    optimizer = optax.sgd(learning_rate=alpha_schedule)
    opt_state = optimizer.init(theta)

    backward = jax.jit(jax.value_and_grad(compute_trajectory_loss))

    def train_step(theta, state):
        loss_val, grads = backward(theta)

        # clip the gradient if needed
        n_grad = jnp.linalg.norm(grads["p"])
        if n_grad > CLIP:
            grads["p"] = grads["p"] / n_grad * CLIP

        updates, new_state = optimizer.update(grads, state)
        return optax.apply_updates(theta, updates), new_state, loss_val

    print("Starting differentiable MPC tuning via gradient descent...")
    print(f"  Initial p: {np.asarray(theta['p'])}\n")

    logger = RunLogger()
    for epoch in range(N_ITER):
        t0 = time.time()
        theta, opt_state, loss_val = train_step(theta, opt_state)
        dt = time.time() - t0
        logger.log(Epoch=f"{epoch:02d}",
                   TaskCost=f"{loss_val:.2f}",
                   Time=f"{dt:.3f}s")

    print(f"\n  Final p: {np.asarray(theta['p'])}") #type: ignore
    return theta


# ============================================================================
# 6. Plotting
# ============================================================================
def plot_before_after(traj_before, traj_after, savepath=None):
    """Plot cart position, pole angle, and control input before vs after tuning."""
    t = np.arange(HORIZON_SIM) * DT

    fig, axes = plt.subplots(3, 1, figsize=(8.5, 7.5), sharex=True)

    # 1. Cart position
    axes[0].plot(t, traj_before["xs"][:, 0],
                 linestyle="--", color="tab:gray",  label="before")
    axes[0].plot(t, traj_after["xs"][:, 0],
                 color="tab:blue", label="after")
    axes[0].axhline(0.0, color="k", linewidth=0.5, linestyle=":")
    axes[0].set_ylabel("Cart position [m]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    # 2. Pendulum angle (target = 0; hanging down = -pi).
    axes[1].plot(t, traj_before["xs"][:, 2],
                 linestyle="--", color="tab:gray", label="before")
    axes[1].plot(t, traj_after["xs"][:, 2],
                 color="tab:orange", label="after")
    axes[1].axhline( 0.0,    color="g", linewidth=0.6, linestyle=":", label="upright")
    axes[1].axhline(-np.pi,  color="r", linewidth=0.6, linestyle=":", label="hanging")
    axes[1].set_ylabel("Pole angle [rad]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best", ncol=2)

    # 3. Control input
    axes[2].plot(t, traj_before["us"][:, 0],
                 linestyle="--", color="tab:gray",  label="before")
    axes[2].plot(t, traj_after["us"][:, 0],
                 color="tab:green", label="after")
    axes[2].axhline(float(U_MAX_MPC[0]), color="r", linewidth=0.6, linestyle=":")
    axes[2].axhline(float(U_MIN_MPC[0]), color="r", linewidth=0.6, linestyle=":")
    axes[2].set_ylabel("Control input [N]")
    axes[2].set_xlabel("Time [s]")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="best")

    fig.suptitle("Closed-loop swing-up: before vs after MPC tuning")
    fig.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
        print(f"Saved comparison plot to {savepath}")
    plt.show()


# ============================================================================
# 7. Main
# ============================================================================
def main():
    # JIT once; reused for both rollouts.
    rollout = jax.jit(simulate_closed_loop)

    # 1. Roll out with the initial DARE-based parameters.
    theta_init = {"p": dare_init_theta(plant)}
    print("Rolling out with initial parameters...")
    traj_before = rollout(theta_init["p"])
    print(f"  Initial task cost: {float(traj_before['cost']):.2f}\n")

    # 2. Tune.
    theta_final = tune_mpc_weights(theta_init)

    # 3. Roll out with the optimised parameters.
    print("\nRolling out with optimised parameters...")
    traj_after = rollout(theta_final["p"]) #type: ignore
    print(f"  Final task cost: {float(traj_after['cost']):.2f}\n")

    # 4. Compare.
    plot_before_after(traj_before, traj_after)


if __name__ == "__main__":
    main()