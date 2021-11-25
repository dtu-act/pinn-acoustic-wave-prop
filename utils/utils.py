# ==============================================================================
# Copyright 2021 Technical University of Denmark
# Author: Nikolas Borrel-Jensen 
#
# All Rights Reserved.
#
# Licensed under the MIT License.
# ==============================================================================
import numpy as np
from models.datastructures import Domain

def calcSourcePositions(grid, domain: Domain):
    
    x0_sources = domain.x0_sources
    Xbounds = domain.Xbounds

    def srcPos1D():
        xmin = Xbounds[0][0]
        xmax = Xbounds[1][0]

        x_data0 = np.asarray(grid[0][0])
        r0 = np.empty(len(x0_sources))

        for i,s1d in enumerate(x0_sources):
            s = s1d[0]
            if s <= 0:
                rx = xmax - np.abs(xmin - s)/2
            else:
                rx = xmin + (xmax - s)/2

            indx_x = (np.abs(x_data0-rx)).argmin()
            r0[i] = x_data0[indx_x]
        
        return r0

    def srcPos2D():
        raise NotImplementedError()        
    
    if domain.spatial_dimension == 1:
        return srcPos1D()
    elif domain.spatial_dimension == 2:
        return srcPos1D()
    else:
        raise NotImplementedError()

def extractSignal(r0,x_data,t_data,p_data):
    tol = 10e-5
    result = np.where(np.abs(x_data.flatten() - r0) <= tol)
    indxs = result[0]
    assert(len(indxs) > 1)

    return p_data.flatten()[indxs],t_data.flatten()[indxs]