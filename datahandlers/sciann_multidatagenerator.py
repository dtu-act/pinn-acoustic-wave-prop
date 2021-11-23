# ==============================================================================
# Copyright 2021 Technical University of Denmark
# Author: Nikolas Borrel-Jensen 
#
# All Rights Reserved.
#
# Licensed under the MIT License.
# ==============================================================================

import numpy as np
from dataclasses import dataclass
from typing import List
import matplotlib.pyplot as plt 

class MultiDataGenerator:
    """ Generates time-dependent collocation grid for training PINNs for multi-domains corresponding to e.g. different source positions.
    # Arguments:
      data_gen: the type of data generator to use. Can be any dimension, but include time dimension.
    """

    def __init__(self,num_grids,data_generator):
        'Initialization'
        self.num_grids = num_grids
        self.gen = data_generator

    def __getitem__(self, indx_domain):
        """indx_domain: index of the domain (0-indexed)"""

        if self.num_grids <= indx_domain:
            raise IndexError()

        offset = int(indx_domain*self.num_sample_domain)

        input_data = self.gen.input_data
        target_data = self.gen.target_data

        # add offset
        indxs, targets = zip(*target_data)
        indxs_offset = [np.add(z,offset) for z in indxs]
        target_data_offset = list(zip(indxs_offset, targets))

        return input_data, target_data_offset
    
    def plot_data(self, block=False):
        return self.gen.plot_data(block)

    @property
    def inputs_data(self):
        x_input = []
        t_input = []

        for i in range(self.num_grids):
            idata, _ = self[i]
            x_input.extend(idata[0])
            t_input.extend(idata[1])

        x_input = np.asarray(x_input)
        t_input = np.asarray(t_input)

        #TODO check
        return np.asarray([x_input.reshape(-1,1),t_input.reshape(-1,1)])

    @property
    def targets_data(self):
        targets_data = [(np.array([], dtype=int),'zeros') for i in range(len(self.gen.targets))]

        for i in range(self.num_grids):
            _, target_data = self[i]
            for j,_ in enumerate(self.gen.targets):
                tdata = targets_data[j]
                tdata_0 = np.append(tdata[0], target_data[j][0])
                tdata_1 = target_data[j][1]
                targets_data[j] = (tdata_0, tdata_1)
                assert(target_data[j][1] == 'zeros')

        return targets_data
    
    @property
    def generator(self):
        return self.gen

    @property
    def num_sample_domain(self):
        return self.gen.num_sample

    @property
    def num_sample_total(self):
        return self.gen.num_sample*self.num_grids

@dataclass
class MultiDataContainer:
    grids: List
    num_grids: int
    offset: int
    
    def __init__(self, grids, offset=0):
        self.spatial_dimension = np.asarray(grids).shape[1] - 1 # return only spatial dim
        self.grids = grids
        self.offset = offset
        self.num_grids = len(grids)

    def __getitem__(self, arg):
        """ 
            Returns dimensional data 
            arg: index of the source/grid
        """

        if self.num_grids <= arg:
            raise IndexError()

        offset = int(arg*self.num_sample_domain) + self.offset

        grid = self.grids[arg]
        target_data = [(np.arange(len(grid[0])) + offset, 'zeros')] # only one target

        return grid, target_data
    
    def plot_data(self, block=False):
        if self.spatial_dimension == 1:
            fig = plt.figure(figsize=(10,8))
            x_data = self.inputs_data[0]
            t_data = self.inputs_data[1]
            plt.scatter(x_data, t_data, s=0.5)
            plt.xlabel('x')
            plt.ylabel('t')
            plt.show(block=block)
        else:
            fig = plt.figure(figsize=(10,8))
            X_data = self.inputs_data[0]
            t_data = self.inputs_data[1]
            plt.plot3D(X_data[0], X_data[1], t_data, s=0.5)
            plt.xlabel('x')
            plt.ylabel('t')
            plt.show(block=block)
        
        return fig

    @property
    def inputs_data(self):
        """ Returns data prepared for training in a flattened array for spatial and temporal dimensions, respectively """

        x_input = []
        t_input = []

        for i in range(self.num_grids):
            idata, _ = self[i]
            x_input.extend(idata[0])
            t_input.extend(idata[1])

        x_input = np.asarray(x_input)
        t_input = np.asarray(t_input)

        return np.asarray([x_input.reshape(-1,1),t_input.reshape(-1,1)])

    @property
    def targets_data(self):
        num_targets = 1
        targets_data = [(np.array([], dtype=int),'zeros') for i in range(num_targets)]

        for i in range(self.num_grids):
            _, target_data = self[i]
            for j in range(num_targets):
                tdata = targets_data[j]
                tdata_0 = np.append(tdata[0], target_data[j][0])
                tdata_1 = target_data[j][1]
                targets_data[j] = (tdata_0, tdata_1)
                assert(target_data[j][1] == 'zeros')

        return targets_data

    @property
    def num_sample_domain(self):
        return len(self.grids[0][0][0])

    @property
    def num_sample_total(self):
        return self.num_sample_domain*self.num_grids