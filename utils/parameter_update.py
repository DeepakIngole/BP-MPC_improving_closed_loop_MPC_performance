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
    def solver(sim):

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

    # generate robust descent solver
    solver = robust_descent_solver(n_models=n_models,n_p=n_p,jit=jit,verbose=verbose)

    def parameter_update(sim,k):

        # get gradient
        max_gradient_error, g_t = solver(sim)

        # get previous parameters
        m_t_1 = sim.psi['m']
        v_t_1 = sim.psi['v']

        # update m and v
        m_t = beta_1*m_t_1 + (1-beta_1)*g_t
        v_t = beta_2*v_t_1 + (1-beta_2)*ca.constpow(g_t,2)

        # compute m_hat and v_hat
        m_hat_t = m_t / (1-beta_1**(k+1))
        v_hat_t = v_t / (1-beta_2**(k+1))

        # adam update
        p_next = sim.p - alpha * ca.rdivide(m_hat_t, ca.sqrt(v_hat_t) + epsilon)

        return {'p':p_next, 'psi':{'m':m_t,'v':v_t}}

    # no initialization for psi
    parameter_init = lambda sim: {'m':ca.DM(*sim.p.shape),'v':ca.DM(*sim.p.shape)}

    return parameter_update,parameter_init

def robust_gradient_descent(rho,eta,n_models,n_p,log=True,jit=False,verbose=False):

    # generate robust descent solver
    solver = robust_descent_solver(n_models=n_models,n_p=n_p,jit=jit,verbose=verbose)

    def parameter_update(sim,k):

        # get direction
        max_gradient_error, d = solver(sim)

        # run GD update
        p_next = gd_rule(p=sim.p,j_p=d,k=k,rho=rho,eta=eta,log=log)

        return {'p':p_next}

    return parameter_update, lambda sim: {}

def gradient_descent(rho,eta=1,log=True):

    def parameter_update(sim,k):

        # run GD update
        p_next = gd_rule(p=sim.p,j_p=sim.j_p,k=k,rho=rho,eta=eta,log=log)

        return {'p':p_next}
    
    return parameter_update, lambda sim: {}

def minibatch_descent(rho,eta=1,log=True,batch_size=1):

    def parameter_update(sim,k):
        
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
    
    def parameter_init(sim):
        return {'j_p':ca.DM.zeros(*sim.j_p.shape)}

    return parameter_update, parameter_init

def average_gradient_descent(rho,eta,log=True):

    def parameter_update(sim,k):

        # average all Jacobians
        j_p = ca.sum2(sim.j_p) / sim.j_p.shape[1]

        # gradient step
        p_next = gd_rule(p=sim.p,j_p=j_p,k=k,rho=rho,eta=eta,log=log)

        return {'p':p_next}

    return parameter_update, lambda sim: {}

def gd_rule(p,j_p,k,rho,eta,log):

    return p - (rho*ca.log(k+2)/(k+1)**eta)*j_p if log else p - (rho/(k+1)**eta)*j_p