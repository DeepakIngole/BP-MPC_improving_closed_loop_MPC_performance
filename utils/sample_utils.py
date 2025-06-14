import numpy as np
import numpy.linalg as LA

# Taken from https://gist.github.com/Bharath2/5cfbf21e3c3f75d3a25d06d8a5f22a7d

def sample_unit_ball(dim = 3,num = 2):
    '''
    uniformly sample a N-dimensional unit UnitBall
    Reference:
      Efficiently sampling vectors and coordinates from the n-sphere and n-ball
      http://compneuro.uwaterloo.ca/files/publications/voelker.2017.pdf
    Input:
        num - no. of samples
        dim - dimensions
    Output:
        uniformly sampled points within N-dimensional unit ball
    '''
    #Sample on a unit N+1 sphere
    u = np.random.normal(0, 1, (num, dim + 2))
    norm = LA.norm(u, axis = -1,keepdims = True)
    u = np.multiply(u/norm, np.random.rand(num,1))
    return u[:,:dim]

# def sample_unit_ball(dim = 3,num = 2):
#     '''
#     uniformly sample a N-dimensional unit UnitBall
#     Reference:
#       Efficiently sampling vectors and coordinates from the n-sphere and n-ball
#       http://compneuro.uwaterloo.ca/files/publications/voelker.2017.pdf
#     Input:
#         num - no. of samples
#         dim - dimensions
#     Output:
#         uniformly sampled points within N-dimensional unit ball
#     '''
#     #Sample on a unit N+1 sphere
#     u = np.random.normal(0, 1, (num, dim + 2))
#     norm = LA.norm(u, axis = -1,keepdims = True)
#     u = u/norm
#     return u[:,:dim]