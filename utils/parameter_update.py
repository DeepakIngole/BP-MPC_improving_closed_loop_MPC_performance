import casadi as ca
import numpy as np
from typing import Callable, Tuple
from src.sim_var import SimVar


def robust_descent_solver(
        n_models:int,
        n_p:int,
        jit:bool=False,
        verbose:bool=False
    ) -> Callable[[SimVar],Tuple[ca.DM,ca.DM]]:
    """
    Creates a robust descent QP solver for parameter update in multi-model settings.

    This function constructs and returns a solver function that, given simulation variables,
    solves a quadratic program (QP) to find the parameter update direction that minimizes
    the worst-case gradient error across multiple models.

    Args:
        n_models (int): Number of models/scenarios considered in the robust descent.
        n_p (int): Number of parameters to be updated.
        jit (bool, optional): If True, enables just-in-time compilation for faster execution. Defaults to False.
        verbose (bool, optional): If True, enables verbose output from the QP solver. Defaults to False.

    Returns:
        Callable[[SimVar], Tuple[ca.DM, ca.DM]]:
            A solver function that takes a SimVar object (with attribute `j_p` representing the gradient matrix)
            and returns a tuple:
                - max_gradient_error (ca.DM): The maximum gradient error found by the QP.
                - gradient (ca.DM): The parameter update direction vector.

    Notes:
        - Requires CasADi (imported as `ca`) for symbolic and numerical optimization.
        - The returned solver expects `sim.j_p` to be compatible with CasADi's DM type and shape.
    """

    # compilation options
    if jit:
        jit_options = {"flags": "-O3", "verbose": False, "compiler": "gcc -Ofast -march=native"}
        options = {"jit": True, "compiler": "shell", "jit_options": jit_options}
    else:
        options = {}

    if not verbose:
        options = options | {'osqp':{'verbose':False}}

    # create optimization variables
    d = ca.SX.sym('d',n_p,1)
    cost = ca.SX.sym('cost',1,1)

    # create constraint functions
    g1 = -ca.repmat(d,n_models,1) + ca.SX.ones(n_models*n_p,1)*cost
    g2 = ca.repmat(d,n_models,1) + ca.SX.ones(n_models*n_p,1)*cost

    # form objective
    f = cost**2

    # form QP solver
    solve_robust_descent_qp = ca.qpsol('S','osqp',{'x':ca.vertcat(cost,d),'f':f,'g':ca.vertcat(g1,g2)},options)

    # wrapper around S
    def solver(sim:SimVar) -> Tuple[ca.DM, ca.DM]:
        """
        Solves a robust descent quadratic program (QP) to compute the maximum gradient error and the gradient vector.

        Args:
            sim (SimVar): Simulation variables containing the gradient matrix `j_p`.

        Returns:
            Tuple[ca.DM, ca.DM]: 
                - max_gradient_error (ca.DM): The maximum gradient error obtained from the QP solution.
                - gradient (ca.DM): The gradient vector obtained from the QP solution.
        """

        # get gradient matrix and form lower-bound
        j_p = ca.reshape(ca.DM(sim.j_p),-1,1)

        # solve
        sol = solve_robust_descent_qp(lbg=ca.vertcat(-j_p,j_p))['x']

        # extract solution
        max_gradient_error, gradient = sol[0], sol[1:0]

        return max_gradient_error, gradient

    return solver


def robust_adam(
        n_models:int,
        n_p:int,
        alpha:float=0.001,
        beta_1:float=0.9,
        beta_2:float=0.999,
        epsilon:float=1e-8,
        jit:bool=False,
        verbose:bool=False
    ) -> Tuple[Callable,Callable]:
    """
    Creates a robust Adam optimizer for parameter updates using gradients from a robust descent solver.
    Args:
        n_models (int): Number of models used in the robust descent solver.
        n_p (int): Number of parameters to optimize.
        alpha (float, optional): Learning rate for the Adam optimizer. Default is 0.001.
        beta_1 (float, optional): Exponential decay rate for the first moment estimates. Default is 0.9.
        beta_2 (float, optional): Exponential decay rate for the second moment estimates. Default is 0.999.
        epsilon (float, optional): Small constant for numerical stability. Default is 1e-8.
        jit (bool, optional): Whether to use just-in-time compilation for the solver. Default is False.
        verbose (bool, optional): Whether to print verbose output during optimization. Default is False.
    Returns:
        Tuple[Callable, Callable]: 
            - parameter_update: A function that performs a parameter update given the simulation state and iteration index.
            - parameter_init: A function that initializes the optimizer's internal state (moments) given the simulation state.
    Notes:
        - The returned `parameter_update` function expects a simulation object (`sim`) and the current iteration index (`k`).
        - The returned `parameter_init` function initializes the first and second moment vectors to zeros with the same shape as the parameters.
    """

    # generate robust descent solver
    solver = robust_descent_solver(n_models=n_models,n_p=n_p,jit=jit,verbose=verbose)

    def parameter_update(sim,k):

        # get gradient
        max_gradient_error, g_t = solver(sim)

        # run Adam update
        p_next, m_t, v_t = adam_rule(p=sim.p,m_t_1=sim.psi['m'],v_t_1=sim.psi['v'],g_t=g_t,
                                    k=k,alpha=alpha,beta_1=beta_1,beta_2=beta_2,epsilon=epsilon)
        
        return {'p':p_next, 'psi':{'m':m_t,'v':v_t}}

    # no initialization for psi
    parameter_init = lambda sim: {'m':ca.DM(*sim.p.shape),'v':ca.DM(*sim.p.shape)}

    return parameter_update,parameter_init


def robust_gradient_descent(rho,eta,n_models,n_p,log=True,jit=False,verbose=False) -> Tuple[Callable, Callable]:
    """
    Creates a robust gradient descent parameter update function and an auxiliary function.

    Args:
        rho (float): Regularization or step size parameter for the gradient descent update.
        eta (float): Learning rate for the gradient descent update.
        n_models (int): Number of models used in the robust descent solver.
        n_p (int): Dimension of the parameter vector to be updated.
        log (bool, optional): If True, enables logging during the gradient descent update. Defaults to True.
        jit (bool, optional): If True, enables JIT compilation for the robust descent solver. Defaults to False.
        verbose (bool, optional): If True, enables verbose output during solver execution. Defaults to False.

    Returns:
        Tuple[Callable, Callable]: 
            - parameter_update: A function that performs a robust gradient descent update on the parameters given 
                a simulation object and time step.
            - A lambda function that takes a simulation object and returns an empty dictionary (placeholder 
                for compatibility).
    """

    # generate robust descent solver
    solver = robust_descent_solver(n_models=n_models,n_p=n_p,jit=jit,verbose=verbose)

    def parameter_update(sim:SimVar,k:int) -> dict:
        """
        Updates the parameter vector using a gradient descent rule based on the current simulation state.

        Args:
            sim: The current simulation object containing the parameter vector and other relevant data.
            k (int): The current iteration or time step.

        Returns:
            dict: A dictionary containing the updated parameter vector under the key 'p'.

        Notes:
            - The function computes the gradient direction using the `solver` function.
            - The parameter update is performed using the `gd_rule` function, which applies a gradient descent step.
            - Additional arguments such as `rho`, `eta`, and `log` are assumed to be available in the enclosing scope.
        """

        # get direction
        max_gradient_error, d = solver(sim)

        # run GD update
        p_next = gd_rule(p=sim.p,j_p=d,k=k,rho=rho,eta=eta,log=log)

        return {'p':p_next}

    return parameter_update, lambda sim: {}


def gradient_descent(rho:float,eta:float=1.0,log:bool=True) -> Tuple[Callable, Callable]:
    """
    Performs gradient descent parameter update for a simulation object.
    Args:
        rho (float): Regularization or step size parameter for the gradient descent update.
        eta (float, optional): Learning rate for the gradient descent update. Defaults to 1.0.
        log (bool, optional): If True, enables logging during the update. Defaults to True.
    Returns:
        tuple: 
            - parameter_update (Callable): A function that updates the simulation parameters using gradient descent.
            - (Callable): An auxiliary function that currently returns an empty dictionary.
    The returned `parameter_update` function expects arguments `(sim, k)`, where:
        sim: An object with attributes `p` (parameters) and `j_p` (gradient of the objective with respect to parameters).
        k: The current iteration or time step.
    The update rule is performed by the `gd_rule` function (not shown here), which applies the gradient descent step.
    """

    def parameter_update(sim:SimVar,k:int) -> dict:
        """
        Updates the parameter vector using a gradient descent rule.

        Args:
            sim: An object containing the current simulation state, including:
                - p: Current parameter vector.
                - j_p: Gradient of the cost function with respect to the parameters.
            k (int): The current iteration or time step.

        Returns:
            dict: A dictionary containing the updated parameter vector with key 'p'.

        Notes:
            This function applies a gradient descent update rule (gd_rule) to the parameter vector.
            Additional arguments such as rho, eta, and log are assumed to be available in the scope.
        """

        # run GD update
        p_next = gd_rule(p=sim.p,j_p=sim.j_p,k=k,rho=rho,eta=eta,log=log)

        return {'p':p_next}
    
    return parameter_update, lambda sim: {}


def minibatch_descent(rho:float,eta:float=1.0,log:bool=True,batch_size:int=1) -> Tuple[Callable, Callable]:
    """
    Implements a minibatch gradient descent parameter update rule.
    Args:
        rho (float): Learning rate or step size for the update.
        eta (float, optional): Scaling factor for the update. Defaults to 1.0.
        log (bool, optional): If True, enables logging during the update. Defaults to True.
        batch_size (int, optional): Number of samples in each minibatch. Defaults to 1.
    Returns:
        tuple: A tuple containing two functions:
            - parameter_update(sim, k): Updates parameters using minibatch gradient descent.
                Args:
                    sim: Simulation or optimization state object containing current parameters and gradients.
                    k (int): Current iteration or step index.
                Returns:
                    dict: Updated parameters and running gradient.
            - parameter_init(sim): Initializes the running gradient accumulator.
                Args:
                    sim: Simulation or optimization state object.
                Returns:
                    dict: Initialized running gradient.
    """

    def parameter_update(sim:SimVar,k:int) -> dict:
        """
        Updates the parameters and running gradient for a simulation at each iteration.
        This function checks if a batch update should be performed based on the current step `k` and the `batch_size`.
        If the batch size is reached, it averages the accumulated gradients, resets the running gradient, and updates
        the parameters using a gradient descent rule. Otherwise, it accumulates the gradient for the next batch update.
        Args:
            sim (SimVar): Simulation variable object containing current parameters, gradients, and running gradient.
            k (int): Current iteration or step index.
        Returns:
            dict: A dictionary containing the updated parameters ('p') and the updated running gradient ('psi').
        """
        
        # check if number of steps has been reached
        if ca.fmod(k+1,batch_size) == 0:

            # construct average gradient
            j_p = (sim.psi['j_p'] + sim.j_p) / batch_size

            # zero the running gradient
            psi = {'j_p':ca.DM.zeros(*j_p.shape)}
            
            # run update
            p = gd_rule(p=sim.p,j_p=j_p,k=k,rho=rho,eta=eta,log=log)

        # else update gradient
        else:
            psi = sim.psi['j_p'] + sim.j_p
            p = sim.p

        return {'p':p,'psi':psi}
    
    def parameter_init(sim:SimVar) -> dict:
        """
        Initializes the parameter dictionary for the simulation.
        Args:
            sim (SimVar): An instance containing simulation variables, including the shape of 'j_p'.
        Returns:
            dict: A dictionary with a single key 'j_p', whose value is a CasADi DM zero matrix matching the shape of sim.j_p.
        """
        return {'j_p':ca.DM.zeros(*sim.j_p.shape)}

    return parameter_update, parameter_init


def average_gradient_descent(rho:float,eta:float,log:bool=True) -> Tuple[Callable, Callable]:
    """
    Creates a parameter update function using average gradient descent.

    Args:
        rho (float): Regularization parameter or step size modifier for the gradient descent.
        eta (float): Learning rate for the gradient descent update.
        log (bool, optional): If True, enables logging during the update. Defaults to True.

    Returns:
        Tuple[Callable, Callable]: 
            - parameter_update (Callable): A function that performs a parameter update using the average of all Jacobians and a gradient descent rule.
            - (Callable): An auxiliary function that currently returns an empty dictionary.

    The returned `parameter_update` function expects:
        sim (SimVar): Simulation variables containing parameters and Jacobians.
        k (int): Current iteration or time step.

    Returns from `parameter_update`:
        dict: Dictionary containing the updated parameter 'p'.
    """

    def parameter_update(sim:SimVar,k:int) -> dict:
        """
        Updates the parameter vector using averaged Jacobians and a gradient descent rule.

        Args:
            sim (SimVar): Simulation variables containing the current parameter vector `p` and Jacobians `j_p`.
            k (int): Current iteration or time step.

        Returns:
            dict: A dictionary containing the updated parameter vector under the key 'p'.

        Notes:
            - The function averages all Jacobians in `sim.j_p`.
            - The parameter update is performed using the `gd_rule` function with the averaged Jacobian and other hyperparameters (`rho`, `eta`, `log`).
        """

        # average all Jacobians
        j_p = ca.sum2(sim.j_p) / sim.j_p.shape[1]

        # gradient step
        p_next = gd_rule(p=sim.p,j_p=j_p,k=k,rho=rho,eta=eta,log=log)

        return {'p':p_next}

    return parameter_update, lambda sim: {}


def gd_rule(p:ca.DM,j_p:ca.DM,k:int,rho:float,eta:float,log:bool) -> ca.DM:
    """
    Applies a gradient descent update rule to the parameter vector.

    Args:
        p (ca.DM): Current parameter vector.
        j_p (ca.DM): Gradient of the objective function with respect to the parameters.
        k (int): Current iteration number (zero-based).
        rho (float): Learning rate scaling factor.
        eta (float): Exponent controlling the learning rate decay.
        log (bool): If True, applies a logarithmic decay to the learning rate; otherwise, uses polynomial decay.

    Returns:
        ca.DM: Updated parameter vector after applying the gradient descent step.
    """
    return p - (rho*ca.log(k+2)/(k+1)**eta)*j_p if log else p - (rho/(k+1)**eta)*j_p

def adam_rule(
        p:ca.DM,
        m_t_1:ca.DM,
        v_t_1:ca.DM,
        g_t:ca.DM,
        k:int,
        alpha:float,
        beta_1:float,
        beta_2:float,
        epsilon:float
    ) -> Tuple[ca.DM, ca.DM, ca.DM]:
    """
    Performs a single update step of the Adam optimization algorithm.

    Args:
        p (ca.DM): Current parameter value.
        m_t_1 (ca.DM): Exponential moving average of past gradients (first moment) at previous step.
        v_t_1 (ca.DM): Exponential moving average of past squared gradients (second moment) at previous step.
        g_t (ca.DM): Current gradient.
        alpha (float): Learning rate.
        beta_1 (float): Exponential decay rate for the first moment estimates.
        beta_2 (float): Exponential decay rate for the second moment estimates.
        epsilon (float): Small constant for numerical stability.

    Returns:
        Tuple[ca.DM, ca.DM, ca.DM]: Tuple containing the updated parameter, first moment, and second moment.
    """

    # update m and v
    m_t = beta_1*m_t_1 + (1-beta_1)*g_t
    v_t = beta_2*v_t_1 + (1-beta_2)*ca.constpow(g_t,2)

    # compute m_hat and v_hat
    m_hat_t = m_t / (1-beta_1**(k+1))
    v_hat_t = v_t / (1-beta_2**(k+1))

    # adam update
    p_next = p - alpha * ca.rdivide(m_hat_t, ca.sqrt(v_hat_t) + epsilon)

    return p_next, m_t, v_t