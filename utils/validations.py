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
import datahandlers.training_data_setup as training
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
    x,t,s,_,_ = models.setupNN_PDE(settings.network.p_nn) # dummy - needed to evaluate/test source defined as functional 
    if source_info.type == SourceType.IC:
        source_f = source_info.source(x,s)

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
    print(f'Number of sources: {domain.num_sources}')
    print('dx = %0.4f' % domain.dx)
    print('dt = %0.4f' % domain.dt)
    print(f'nx_tot = {domain.nx}')
    print(f'nt_tot = {domain.nt}')
    print(f'tot = {domain.nx*domain.nt}')
    print('------------')
    print('')

def printSettings(path_to_settings):
    f = open(path_to_settings, 'r')
    content = f.read()
    print(content)
    f.close()

def validateData(settings: Settings):
    training_data_path = settings.dirs.data_path
    
    xmin = settings.domain.xmin
    xmax = settings.domain.xmax
    tmin = settings.domain.tmin
    tmax = settings.domain.tmax
    
    x0_sources = settings.domain.x0_sources

    c = settings.physics.c
    fmax = settings.physics.fmax    
    rho = settings.physics.rho
    sigma0 = settings.physics.sigma0

    dt_load,dx_load,x0_sources_load,physics_load,tmax_load = training.loadAttrFromH5(training_data_path)

    assert physics_load.c == c, f'Settings and loaded data differs: c={c} !=  c_loaded={physics_load.c}'
    assert abs(physics_load.fmax - fmax) < 0.001, f'Settings and loaded data differs: fmax={fmax} !=  fmax_loaded={physics_load.fmax}'
    assert abs(physics_load.sigma0 - sigma0) < 0.001, f'Settings and loaded data differs: sigma0={sigma0} !=  sigma0_loaded={physics_load.sigma0}'
    assert abs(physics_load.rho - rho) < 0.001, f'Settings and loaded data differs: rho={rho} !=  rho_loaded={physics_load.rho}'
    assert len(x0_sources_load) == len(x0_sources) and (x0_sources_load == x0_sources).all(), f'Settings and loaded data differs: x0={x0_sources} !=  x0_loaded={x0_sources_load}'