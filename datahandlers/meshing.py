# ==============================================================================
# Copyright 2021 Technical University of Denmark
# Author: Nikolas Borrel-Jensen 
#
# All Rights Reserved.
#
# Licensed under the MIT License.
# ==============================================================================

from collections import namedtuple
import numpy as np
import datahandlers.sciann_multidatagenerator as mdg
import datahandlers.sciann_datagenerator as dg
from models.datastructures import Domain

TargetIndexes1D = namedtuple('target_indexes_1D',['domain','ic','bc_left','bc_right','bc','all'])
TargetIndexes2D = namedtuple('target_indexes_2D',['domain','ic','bc_left','bc_right','bc_bot','bc_top','bc','all'])

def generateNonUniformMesh(domain: Domain, hypercube=True):
    """ data DIMENSIONS
            [i][][]: i source index
            [][j][]: j = 0,1, where 0 is grid data and 1 is targets (for source i)
            [][][k]: k = 0,1, where 0 is x dim and 1 is t dim

        data.inputs_data returns grouped array with [x,t] data for each source:
            [ [[x_0,x_1,...,],[t_0,t_1,...]], ...]
    """
    if domain.spatial_dimension == 1:
        targets = ["domain", "ic", "bc-left", "bc-right", "bc", "all"]
        target_indxs = TargetIndexes1D(domain=0,ic=1,bc_left=2,bc_right=3,bc=4,all=5)
        data_gen = dg.DataGeneratorXT(
            X = [domain.Xbounds[0][0], domain.Xbounds[1][0]],
            T = [0,domain.tmax],
            bc_points_p = domain.bc_points_p,
            ic_points_p = domain.ic_points_p,
            targets = targets,
            num_sample = domain.nX[0]*domain.nt,
            hypercube=hypercube
        )
    elif domain.spatial_dimension == 2:
        targets = ["domain", "ic", "bc-left", "bc-right", 'bc-bot', 'bc-top', "bc", "all"]
        target_indxs = TargetIndexes2D(domain=0,ic=1,bc_left=2,bc_right=3,bc_bot=4,bc_top=5,bc=6,all=7)
        data_gen = dg.DataGeneratorXYT(
            X = [domain.Xbounds[0][0], domain.Xbounds[1][0]],
            Y = [domain.Xbounds[0][1], domain.Xbounds[1][1]],
            T = [0,domain.tmax],
            bc_points_p = domain.bc_points_p,
            ic_points_p = domain.ic_points_p,
            targets = targets,
            num_sample = domain.nX[0]*domain.nX[1]*domain.nt,
            hypercube=hypercube
        )
    else:
        raise NotImplementedError()

    data = mdg.MultiDataGenerator(
        num_grids = domain.num_sources,
        data_generator = data_gen            
    )
    
    # create an array with number of entries corresponding to the number of source posiion
    x0_input_data = np.asarray([[x0,]*len(data[i][0][0]) for i,x0 in enumerate(domain.x0_sources)])

    return data, x0_input_data, target_indxs

def generateUniformMesh(domain: Domain, offset=0):
    """
        [i][]: i source index
        [][k]: k = 0,1, where 0 is X data and 1 is t data
    """
    
    if domain.spatial_dimension == 1:
        grid = np.meshgrid(
            np.linspace(domain.Xbounds[0][0], domain.Xbounds[1][0], domain.nX[0]),
            np.linspace(0, domain.tmax, domain.nt))
        # matrix indexes: (t, x) with size nt X nx
    elif domain.spatial_dimension == 2:
        x,y,t = np.meshgrid(
            np.linspace(domain.Xbounds[0][0], domain.Xbounds[1][0], domain.nX[0]),
            np.linspace(domain.Xbounds[0][1], domain.Xbounds[1][1], domain.nX[1]), 
            np.linspace(0, domain.tmax, domain.nt))
            
        # transpose to get matrix indexes: (t, x, y) with size nt X nx X ny
        grid = [np.transpose(x),np.transpose(y),np.transpose(t)]
    else:
        raise NotImplementedError()

    x0_data = np.asarray([[x0,]*len(grid[0]) for i,x0 in enumerate(domain.x0_sources)])

    return grid, x0_data