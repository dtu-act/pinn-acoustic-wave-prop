# ==============================================================================
# Copyright 2021 Technical University of Denmark
# Author: Nikolas Borrel-Jensen 
#
# All Rights Reserved.
#
# Licensed under the MIT License.
# ==============================================================================
import numpy as np

def calcSourcePositions(data,x0_sources):
    r0 = np.empty(len(x0_sources))

    x_data0 = np.asarray(data[0][0][0])
    for i,s in enumerate(x0_sources):    
        index = (np.abs(x_data0-s)).argmin()
        r0[i] = x_data0[index]
    
    return r0

def extractSignal(r0,x_data,t_data,p_data):
    tol = 10e-5
    result = np.where(np.abs(x_data.flatten() - r0) <= tol)
    indxs = result[0]
    assert(len(indxs) > 1)

    return p_data.flatten()[indxs],t_data.flatten()[indxs]