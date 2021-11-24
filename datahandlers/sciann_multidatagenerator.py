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