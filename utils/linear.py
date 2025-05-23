import numpy as np
import casadi as ca
from scipy.linalg import expm,eig
from typing import Tuple

def poles_to_linear_sys(poles:np.array,Ts:int=0.5) -> Tuple[ca.SX,ca.SX]:
    """
    Constructs a discrete-time linear state-space system (A, B) in controllable canonical form from a given set of continuous-time poles.
    Args:
        poles (np.array): Array of continuous-time poles (eigenvalues) specifying the desired system dynamics.
        Ts (int, optional): Sampling time for discretization. Defaults to 0.5.
    Returns:
        Tuple[ca.SX, ca.SX, np.ndarray]: 
            - A (ca.SX): Discrete-time state transition matrix.
            - B (ca.SX): Discrete-time input matrix.
            - eig_a (np.ndarray): Absolute values of the eigenvalues of the discretized A matrix.
    Raises:
        AssertionError: If the eigenvalues of the constructed continuous-time A matrix do not match the provided poles.
    Notes:
        - The function constructs the continuous-time system in controllable canonical form, then discretizes it using the matrix exponential.
        - Requires CasADi (`ca`) and NumPy (`np`) libraries.
    """

    # get number of states
    n_x = poles.shape[0]
    
    # put ones in the off-diagonal of A
    A_cont = np.diag(np.ones(n_x-1),k=1)

    # substitute last row of A with characteristic polynomial
    A_cont[-1,:] = -np.array(np.flip(np.poly(poles)[1:]))

    # B matrix in controllable canonical form
    B_cont = ca.vcat([ca.DM(n_x-1,1),1])

    # check eigenvalues of A_cont
    eig_a_cont = eig(A_cont)[0]
    
    # check that poles match
    assert np.allclose(np.sort(eig_a_cont),np.sort(poles),rtol=1e-12), 'Poles do not match'

    # Euler
    # A_euler = ca.SX.eye(n_x) + Ts*ca.SX(A_cont)
    # B_euler = Ts*B_cont

    # second order
    # A_second_order = ca.SX.eye(n_x) + Ts*ca.SX(A_cont) + Ts**2/2*ca.SX(A_cont@A_cont)

    # discretize
    A = ca.cse(ca.sparsify(ca.SX(expm(Ts*A_cont))))
    B = ca.cse(ca.sparsify(ca.SX(ca.pinv(A_cont)@(A-ca.SX.eye(n_x))@B_cont)))

    # check eigenvalues of A
    eig_a = eig(expm(Ts*A_cont))[0]

    return A,B,np.absolute(eig_a)