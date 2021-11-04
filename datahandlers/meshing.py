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
from models.datastructures import Domain

TargetIndexes = namedtuple('target_indexes',['domain','ic','bc_left','bc_right','bc','point_source','all'])

def generateNonUniformMesh(domain: Domain, hypercube=True):
    """ data DIMENSIONS
            [i][][]: i source index
            [][j][]: j = 0,1, where 0 is grid data and 1 is targets (for source i)
            [][][k]: k = 0,1, where 0 is x dim and 1 is t dim

        data.inputs_data returns grouped array with [x,t] data for each source:
            [ [[x_0,x_1,...,],[t_0,t_1,...]], ...]
    """
    
    targets = ["domain", "ic", "bc-left", "bc-right", "bc", "all"]
    target_indxs = TargetIndexes(0,1,2,3,4,None,5)

    data = mdg.MultiDataGeneratorXT(
        X = domain.X,
        T = domain.T,
        bc_points_p = domain.bc_points_p,
        ic_points_p = domain.ic_points_p,
        targets = targets,
        num_sample_x = domain.nx,
        num_sample_t = domain.nt,
        num_domains = domain.num_sources,
        hypercube=hypercube
    )
    
    x0_input_data = np.asarray([[x0,]*len(data[i][0][0]) for i,x0 in enumerate(domain.x0_sources)])

    return data, x0_input_data, target_indxs

def generateUniformMesh(domain: Domain, offset=0):
    """ data DIMENSIONS
            [i][]: i source index
            [][k]: k = 0,1, where 0 is x data and 1 is t data
    """

    xt_grid = []

    i=0    
    while i < domain.num_sources:
        x_data, t_data = np.meshgrid(
            np.linspace(domain.X[0], domain.X[1], domain.nx), 
            np.linspace(domain.T[0], domain.T[1], domain.nt))
        xt_grid.append([x_data.reshape(-1,1), t_data.reshape(-1,1)])
        i = i+1

    data = mdg.MultiDataContainerXT(xt_grid, offset=offset)
    x0_input_data = np.asarray([[x0,]*len(data[i][0][0]) for i,x0 in enumerate(domain.x0_sources)])

    target_indxs = TargetIndexes(None,None,None,None,None,None,0)

    return data, x0_input_data, target_indxs