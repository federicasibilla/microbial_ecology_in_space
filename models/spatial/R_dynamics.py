
"""
R_dynamics.py: file containing the functions to calculate the reaction equation of resources 

CONTAINS: - f: reaction part of the RD equation, calculates uptake and production, given the 
               concentration at each site

"""

import numpy as np

#-----------------------------------------------------------------------------------------------------------
# f vectorial function to calculate the sources, given the nxnxn_r concentrations matrix

def f(R, N, param, mat):

    """
    R:     matrix, nxnxn_r, state of resources
    N:     matrix, nxnxn_s, state of population
    param: dictionary, parameters
    mat:   dictionary, matrices

    RETURNS inn-upp: matrix, nxnxn_r, reaction part of RD equation 
            upp:     matrix, nxnxn_r, consumption
            inn:     matrix, nxnxn_r, production
    """
    species = np.argmax(N, axis=2)
    
    # Calculate MM at each site
    upp = R / (R + 1)
    uptake_species = mat['uptake'][species]
    up = upp * uptake_species
    
    # Calculate production
    spec_met_species = mat['spec_met'][species]
    l_w = param['l'] * param['w']
    met_transposed = mat['met'].T
    
    # Compute met_grid without explicit tiling
    met_grid = met_transposed[np.newaxis, np.newaxis, :, :] * spec_met_species[:, :, np.newaxis, :]
    
    # Sum along the met_dim axis
    col_sums = np.sum(met_grid, axis=-2, keepdims=True)
    col_sums[col_sums == 0] = 1e-10  # Avoid division by zero
    
    met_grid_normalized = met_grid / col_sums
    
    # Calculate inn using einsum for efficient summation
    inn = np.einsum('ijk,ijkl->ijl', up * l_w, met_grid_normalized) / param['w']
    
    return inn - up, upp, inn

