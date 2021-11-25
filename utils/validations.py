# ==============================================================================
# Copyright 2021 Technical University of Denmark
# Author: Nikolas Borrel-Jensen 
#
# All Rights Reserved.
#
# Licensed under the MIT License.
# ==============================================================================
import numpy as np
import matplotlib.pyplot as plt
import os

from numpy.testing._private.utils import assert_equal
import models.sciann_models as models
from models.datastructures import Domain, SourceType
import datahandlers.data_reader_writer as rw
from setup.settings import Settings

def plotGridSource(data, settings: Settings, tag=''):
    x0_sources = settings.domain.x0_sources
    source_info = settings.domain.source
    figs_dir = settings.dirs.figs_dir

    fig = data.plot_data(block=False)
    
    if figs_dir != None:
        path_plot = os.path.join(figs_dir, f'grid_distribution{tag}.png')        
        fig.savefig(path_plot)

    # plot source/IC
    funcs = models.setupNN_PDE(settings) # dummy - needed to evaluate/test source defined as functional 
    if source_info.type == SourceType.IC:
        source_f = source_info.source(funcs.x,funcs.x0)

        plt.figure(figsize=(10, 8))
        for i,x0 in enumerate(x0_sources):
            x_data_i = np.asarray(data[i][0][0])
            x0_data_i = np.asarray([x0,]*len(x_data_i))
            p_i = source_f.eval([x_data_i, x0_data_i])

            plt.plot(x_data_i, p_i, '.', label=f"IC x0={x0}")
            plt.xlabel("x")
            plt.ylabel("t")
        plt.legend(loc='upper left')

        if figs_dir != None:
            path = os.path.join(figs_dir, f"sources_ic{tag}.png")
            plt.savefig(path)
    else:
        raise NotImplementedError()

def printGridStats(domain: Domain):
    print('')
    print('------------')
    print('PINN GRID stats')
    print(f'spatial dim: {domain.spatial_dimension}')
    print(f'Number of sources: {domain.num_sources}')
    print('dx = %0.4f' % domain.dx)
    print('dt = %0.4f' % domain.dt)
    print(f'nX = {domain.nX}')
    print(f'nt_tot = {domain.nt}')
    if domain.spatial_dimension == 1:
        tot_points = domain.nX[0]*domain.nt
    elif domain.spatial_dimension == 2:
        tot_points = domain.nX[0]*domain.nX[1]*domain.nt
    else:
        tot_points = domain.nX[0]*domain.nX[1]*domain.nX[2]*domain.nt
    print(f'total points = {tot_points}')
    print('------------')
    print('')

def printSettings(path_to_settings):
    f = open(path_to_settings, 'r')
    content = f.read()
    print(content)
    f.close()

def validateData(settings: Settings):
    ref_data_path = settings.dirs.data_path    
    x0_sources = settings.domain.x0_sources

    c = settings.physics.c
    fmax = settings.physics.fmax    
    rho = settings.physics.rho
    sigma0 = settings.physics.sigma0

    _,_,x0_sources_load,physics_load,_ = rw.loadAttrFromH5(ref_data_path)

    assert physics_load.c == c, f'Settings and loaded data differs: c={c} !=  c_loaded={physics_load.c}'
    assert abs(physics_load.fmax - fmax) < 0.001, f'Settings and loaded data differs: fmax={fmax} !=  fmax_loaded={physics_load.fmax}'
    assert abs(physics_load.sigma0 - sigma0) < 0.001, f'Settings and loaded data differs: sigma0={sigma0} !=  sigma0_loaded={physics_load.sigma0}'
    assert abs(physics_load.rho - rho) < 0.001, f'Settings and loaded data differs: rho={rho} !=  rho_loaded={physics_load.rho}'

    assert len(x0_sources_load) == len(x0_sources) and (x0_sources_load == np.asarray(x0_sources)).all(), f'Settings and loaded data differs: x0={x0_sources} !=  x0_loaded={x0_sources_load}'