# ==============================================================================
# Copyright 2021 Technical University of Denmark
# Author: Nikolas Borrel-Jensen 
#
# All Rights Reserved.
#
# Licensed under the MIT License.
# ==============================================================================

import datahandlers.sciann_datagenerator as dg
import numpy as np
from dataclasses import dataclass
from typing import List
import matplotlib.pyplot as plt 

class MultiDataGeneratorXT:
    """ Generates 1D time-dependent collocation grid for training PINNs for multi-domains corresponding to e.g. different source positions.
    # Arguments:
      X: [X0, X1]
      T: [T0, T1]
      targets: list and type of targets you wish to impose on PINNs. 
          ('domain', 'ic', 'bc-left', 'bc-right', 'all')
      num_sample_x: total number of collocation points in x dimension. 
      num_sample_t: total number of collocation points in t dimension.
      bc_points_p: fraction of total number of points to use at the boundaries.
      ic_points_p: fraction of total number of points to use at t=0 (initial condition).
      num_domains: number of domains. E.g., equal to the number of sources to train.
      hypercube: if True, use Latin Hypercube sampling.
      logT: generate random samples logarithmic in time.
    """

    def __init__(self,
                 X=[0., 1.],
                 T=[0., 1.],
                 targets=['domain', 'ic', 'bc-left', 'bc-right'],
                 num_sample_x=100,
                 num_sample_t=100,
                 bc_points_p=0.25,
                 ic_points_p=0.25,
                 num_domains=1,
                 hypercube=True,
                 logT=False):
        'Initialization'
        self.num_domains = num_domains

        self.gen = dg.DataGeneratorXT(
            X=X,
            T=T,
            bc_points_p=bc_points_p,
            ic_points_p=ic_points_p,
            targets=targets,
            num_sample_x=num_sample_x,
            num_sample_t=num_sample_t,
            hypercube=hypercube,
            logT=logT
        )

    def __getitem__(self, indx_domain):
        """indx_domain: index of the domain (0-indexed)"""

        if self.num_domains <= indx_domain:
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

        for i in range(self.num_domains):
            idata, _ = self[i]
            x_input.extend(idata[0])
            t_input.extend(idata[1])

        x_input = np.asarray(x_input)
        t_input = np.asarray(t_input)

        return np.asarray([x_input.reshape(-1,1),t_input.reshape(-1,1)])

    # @property
    # def inputs_data(self):
    #     inputs_data = np.tile(self.gen.input_data, self.num_domains)
    #     return inputs_data

    @property
    def targets_data(self):
        targets_data = [(np.array([], dtype=int),'zeros') for i in range(len(self.gen.targets))]

        for i in range(self.num_domains):
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
        return self.gen.num_sample*self.num_domains

@dataclass
class MultiDataContainerXT:    
    xt_grids: List[ List[List[List[float]] ] ]
    num_domains: int
    offset: int
    
    def __init__(self, xt_grids, offset=0):
        self.xt_grids = xt_grids
        self.offset = offset
        self.num_domains = len(xt_grids)        

    def __getitem__(self, arg):
        """arg is the index of the domain"""

        if self.num_domains <= arg:
            raise IndexError()

        offset = int(arg*self.num_sample_domain) + self.offset

        input_data = self.xt_grids[arg]
        target_data = [(np.arange(len(input_data[0])) + offset, 'zeros')] # only one target

        return input_data, target_data
    
    def plot_data(self, block=False):
        fig = plt.figure(figsize=(10,8))
        x_data = self.inputs_data[0]
        t_data = self.inputs_data[1]
        plt.scatter(x_data, t_data, s=0.5)
        plt.xlabel('x')
        plt.ylabel('t')
        plt.show(block=block)
        return fig

    @property
    def inputs_data(self):
        x_input = []
        t_input = []

        for i in range(self.num_domains):
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

        for i in range(self.num_domains):
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
        return len(self.xt_grids[0][0][0])

    @property
    def num_sample_total(self):
        return self.num_sample_domain*self.num_domains