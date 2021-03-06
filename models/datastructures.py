# ==============================================================================
# Copyright 2021 Technical University of Denmark
# Author: Nikolas Borrel-Jensen 
#
# All Rights Reserved.
#
# Licensed under the MIT License.
# ==============================================================================
from collections import namedtuple
from dataclasses import dataclass
from enum import Enum
import os
from pathlib import Path
import shutil
from typing import Callable, List
import models.sources as sources

import numpy as np

SciannFunctionals = namedtuple('target_indexes',['x','y','t','x0','p','v'])
Accumulators = namedtuple('accumulators',['phi','psi0','psi1'])

class BoundaryType(Enum):
    DIRICHLET = 1
    NEUMANN = 2
    IMPEDANCE_FREQ_DEP = 3
    IMPEDANCE_FREQ_INDEP = 4

class SourceType(Enum):
    IC = 1
    INJECTION = 2

class LossType(Enum):
    DATA = 'data_loss'
    PINN = 'pinn_loss'
    PINN_DATA = 'pinn_data_loss'

@dataclass
class LossPenalties:
    pde: float
    bc: float
    data: float
    ic: float = None
    ade: float = None

@dataclass
class SourceInfo:
    type: SourceType
    mu: float = 0
    sigma0: float = None
    source: Callable = None

    def __init__(self, type, sigma0: float, spatial_dim: int):
        self.type = type
        self.sigma0 = sigma0
        self.source = sources.sciann_gaussianIC(sigma0, spatial_dim)

@dataclass
class FrequencyDependentImpedance:
    Yinf: float
    A: List[float]
    B: List[float]
    C: List[float]
    lambdas: List[float]
    alpha: List[float]
    beta: List[float]

@dataclass
class BoundaryCondition:
    type: BoundaryType
    xi: float = None # specific acoustic impedance
    p: float = None   # pressure at boundary
    v: float = None   # velocity at boundary
    impedance_data: FrequencyDependentImpedance = None

    def __init__(self, type, p: float=None, v: float=None, impedance_data=None, xi: float=None):
        self.type = type
        if type == BoundaryType.DIRICHLET:
            if p == None:
                raise Exception('p not set')
            self.p = p
        elif type == BoundaryType.NEUMANN:
            if v == None:
                raise Exception('v not set')
            self.v = v
        elif type == BoundaryType.IMPEDANCE_FREQ_INDEP:
            if xi == None:
                raise Exception('xi not set')
            self.xi = xi
        elif type == BoundaryType.IMPEDANCE_FREQ_DEP:
            if impedance_data == None:
                raise Exception('impedance_data not set')
            self.impedance_data = impedance_data        
        else:
            raise NotImplementedError()

@dataclass
class InputOutputDirs:
    id: str
    id_dir: str
    figs_dir: str
    models_dir: str
    transfer_models_dir: str
    plot_graph_path: str
    data_dir: str
    plot_graph_path: str
    data_path: str

    def __init__(self,settings_dict,base_dir=None):        
        if base_dir == None:
            base_dir = settings_dict['base_dir']
            
        self.id = settings_dict['id']
        self.id_dir = os.path.join(base_dir, "results", self.id)
        self.figs_dir = os.path.join(self.id_dir, "figs")
        self.models_dir = os.path.join(self.id_dir, "models")
        self.data_dir = os.path.join(base_dir, "reference_data")
        self.transfer_models_dir = os.path.join(base_dir, "trained_models")        
        self.data_path = os.path.join(self.data_dir, settings_dict['data_filename'])

        self.plot_graph_path = os.path.join(self.models_dir, f'{LossType.PINN}', 'network.png')

    def createDirs(self, delete_existing=False):
        if delete_existing and Path(self.id_dir).exists():
            shutil.rmtree(self.id_dir)
        
        Path(self.figs_dir).mkdir(parents=True, exist_ok=True)
        Path(self.models_dir).mkdir(parents=True, exist_ok=True)

@dataclass
class TransferLearning:
    boundary_cond: BoundaryCondition
    model_dir: str
    trainable: bool

@dataclass
class Physics:
    sigma0: float
    fmax: float
    c: float
    c_phys: float
    rho: float   

@dataclass
class Domain:
    boundary_cond: BoundaryCondition

    spatial_dimension: int
    Xbounds: List[List[float]]
    tmax: float
    ppw: float

    dt: float
    dx: float

    nX: List[List[int]]
    nt: int

    source: SourceInfo
    x0_sources: List[List[float]]

    ic_points_p: float
    bc_points_p: float

    def __init__(self, Xbounds, tmax, ppw, dt, dx, boundary_cond, sigma0, x0_sources, ic_points_p, bc_points_p):
        assert(len(Xbounds[0]) == len(Xbounds[1]))
        
        if len(Xbounds) > 2:
            raise NotImplementedError()

        self.spatial_dimension = np.asarray(Xbounds).shape[1]
        self.Xbounds = Xbounds
        self.tmax = tmax
        self.ppw = ppw
        self.dt = dt
        self.dx = dx
        self.boundary_cond = boundary_cond
        self.source = SourceInfo(SourceType.IC, sigma0, self.spatial_dimension)
        self.x0_sources = x0_sources
        self.ic_points_p = ic_points_p
        self.bc_points_p = bc_points_p
                
        self.nX = ((np.asarray(Xbounds[1])-np.asarray(Xbounds[0]))/dx).astype(int) # number of spatial points
        self.nt = int(tmax/dt) # number of temporal steps

    @property
    def num_sources(self) -> int:
        return len(self.x0_sources)

@dataclass
class ADENeuralNetwork:
    activation: str
    num_layers: int
    num_neurons: int
    accumulator_norm: List[float] # renamed from accumulator_factors
    weights: LossPenalties

@dataclass
class PressureNeuralNetwork:
    activation: str
    num_layers: int
    num_neurons: int
    weights: LossPenalties

@dataclass
class NetworkSettings:
    epochs: int
    stop_loss_value: float

    batch_size: int
    learning_rate: float
    optimizer: str

    p_nn: PressureNeuralNetwork
    ade_nn: ADENeuralNetwork