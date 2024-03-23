""" These are funcitons I built using numba cuda functions"""

import numpy as np
from numba import cuda 
import cupy as cp

@cuda.jit
def sparse_coordinate_matmul(x, y, output, coord_list):
    
    pos = cuda.grid(1)
    
    if pos < coord_list.shape[0]:
        
        tmp = 0.
        
        for k in range(x.shape[1]):
            
            tmp += x[coord_list[pos][0], k]*y[k, coord_list[pos][1]]
        
        output[pos] = tmp
        
@cuda.jit
def full_coordinate_matmul(x, y, output, coord_list):
    
    pos = cuda.grid(1)
    
    if pos < coord_list.shape[0]:
        
        tmp = 0.
        
        for k in range(x.shape[1]):
            
            tmp += x[coord_list[pos][0], k]*y[k, coord_list[pos][1]]
        
        output[pos] = tmp