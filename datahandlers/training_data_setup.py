# ==============================================================================
# Copyright 2021 Technical University of Denmark
# Author: Nikolas Borrel-Jensen 
#
# All Rights Reserved.
#
# Licensed under the MIT License.
# ==============================================================================
import numpy as np
import os
import tensorflow as tf
import h5py
from pathlib import Path
from datahandlers.meshing import TargetIndexes
from datahandlers.sciann_datagenerator import DataGeneratorXT

from models.datastructures import Accumulators, Domain, BoundaryType, SciannFunctionals, Physics
import models.sciann_models as models
from setup.settings import Settings

def loadModel(settings: Settings, funcs: SciannFunctionals, accs: Accumulators, data: DataGeneratorXT, target_indxs: TargetIndexes):
    tl = settings.transfer_learning

    m_pinn = models.setupPinnModels(settings, funcs, accs, boundary_cond_override=tl.boundary_cond)
    targets_pinn = models.setupPinnTargetsTrain(data, target_indxs, tl.boundary_cond)

    checkpoint_path = os.path.join(settings.dirs.transfer_models_dir, tl.model_dir)
    latest = tf.train.latest_checkpoint(checkpoint_path)
    if latest == None:
        raise FileNotFoundError(f'Weights not found: %s', checkpoint_path)
    m_pinn.load_weights(latest)

    m_pinn.model.trainable = tl.trainable
        
    m_pinn.compile()
    m_pinn.summary()

    return m_pinn, targets_pinn

def loadAttrFromH5(path_data):
        """ Load attributes from simulation data
            https://www.pythonforthelab.com/blog/how-to-use-hdf5-files-in-python/
        """

        with h5py.File(path_data, 'r') as f:
            dt = f.attrs['dt'][0]
            dx = f.attrs['dx'][0]
            c = f.attrs['c'][0]
            rho = f.attrs['rho'][0]
            sigma0 = f.attrs['sigma0'][0]
            fmax = f.attrs['fmax'][0]

            x0_sources = f['x0_sources'][()].flatten()
            tmax = f['t'][-1]

            physics_settings = Physics(sigma0,fmax,c,343,rho)

        return dt,dx,x0_sources,physics_settings,tmax
        
def loadDataFromH5(path_data, tmax=None):
    """ tmax should be given as normalized, i.e. with speed of sound 1 m/s
        The time vector is returned in physical space
    """

    xt_grid = []
    p_data = []
    v_data = []
    acc_l_data = []
    acc_r_data = []

    with h5py.File(path_data, 'r') as f:
        dt = f.attrs['dt'][0]
        x0_sources = f['x0_sources'][()]
        x = f['x'][()]
        t = f['t'][()]
        
        dt_norm = dt

        if tmax == None:
            ilast = len(t)-1            
        else:
            ilist = [i for i, n in enumerate(t) if abs(n - tmax) < dt_norm/2]
            if not ilist:
                raise Exception('t_max exceeds simulation data running time')
            ilast = ilist[0]

        t = t[:ilast]        
        
        x_data, t_data = np.meshgrid(x, t)

        i=0
        while i < len(x0_sources):
            xt_grid.append([x_data.reshape(-1,1), t_data.reshape(-1,1)])
            p = np.asarray(f[f'p{i}'][:ilast,:]).reshape(-1,1).tolist()            
            p_data.append(p)

            if f'v{i}' in f.keys():
                v = np.asarray(f[f'v{i}'][:ilast,:]).reshape(-1,1).tolist()
                v_data.append(v)

            if f'acc_l{i}' in f.keys() and f'acc_r{i}' in f.keys():
                acc_l = f[f'acc_l{i}'][:,:ilast]
                acc_r = f[f'acc_r{i}'][:,:ilast]
                acc_l_data.append(acc_l)
                acc_r_data.append(acc_r)

            i = i+1
        
        p_data = np.asarray(p_data)
        v_data = np.asarray(v_data)

    return xt_grid,p_data,v_data,x0_sources,acc_l_data,acc_r_data

def writeDataToHDF5(x, t, p_sources, domain: Domain, physics: Physics, path_file: str):
    """ Write data as HDF5 format
    """
    """ Ex: writeHDF5([1,2,3],[0.1,0.2,0.3],data,[-0.2,0.0,0.2],0.1,1.0,343,0.1,2000,'test1.h5')
    """
    x0_sources = domain.x0_sources
    dt = domain.dt
    dx = domain.dx
    c = physics.c
    rho = physics.rho
    sigma0 = physics.sigma0
    fmax = physics.fmax

    assert(len(p_sources) == len(x0_sources))
    
    if Path(path_file).exists():
        os.remove(path_file)

    with h5py.File(path_file, 'w') as f:
        f.create_dataset('x0_sources', data=x0_sources)
        f.create_dataset('x', data=x)
        f.create_dataset('t', data=t)

        for i in range(0,len(p_sources)):
            
            data_p = p_sources[i].reshape(len(t),len(x))
            f.create_dataset(f'p{i}', data=data_p)
        
        # wrap in array to be consistent with data generated from Matlab
        f.attrs['dt'] = [dt]
        f.attrs['dx'] = [dx]
        f.attrs['c'] = [c]
        f.attrs['rho'] = [rho]
        f.attrs['sigma0'] = [sigma0]
        f.attrs['fmax'] = [fmax]
        
        print(list(f.keys()))
        print(list(f.attrs))
    
def WE1D(x_mesh, t_mesh, x0, boundary_cond, c, sigma0, num_reflections=4):
    """ Analytical solution with Dirichlet or Neumann boundaries
        Model dirichlet/neuman conditions with arbitrarily many reflections
    """
    assert(boundary_cond == BoundaryType.DIRICHLET or boundary_cond == BoundaryType.NEUMANN)

    amp_sign = 1

    xmin = np.min(x_mesh)
    xmax = np.max(x_mesh)
    
    L = ( xmax - xmin )

    # initial wave solution (no reflections)
    p = 0.5*np.exp(-0.5*((x_mesh-x0 - c*t_mesh)/sigma0)**2) + \
        0.5*np.exp(-0.5*((x_mesh-x0 + c*t_mesh)/sigma0)**2)

    if num_reflections <= 0:
        return p

    # calculate starting positions for reflected waves
    x0_rel = (x0 - xmin) / L # relative position

    for i in range(num_reflections):
        if np.mod(i,2) == 0:
            x0_min = xmin - (i)*(L) - L*x0_rel     # x0 for positive travelling wave
            x0_max = xmax + (i+1)*(L) - L*x0_rel   # x0 for negative travelling wave
        else:
            x0_min = xmin - (i)*(L) - (L - L*x0_rel)  # x0 for positive travelling wave
            x0_max = xmax + (i)*(L) + L*x0_rel        # x0 for negative travelling wave

        if boundary_cond == BoundaryType.DIRICHLET:
            amp_sign = -1*amp_sign

                         
        p_pos = amp_sign*0.5*np.exp(-0.5*((x_mesh-x0_min - c*t_mesh)/sigma0)**2)
        p_neg = amp_sign*0.5*np.exp(-0.5*((x_mesh-x0_max + c*t_mesh)/sigma0)**2)
        p = p + p_pos + p_neg
            
    return p

def generateSolutionData1D(xt_grids, x0_sources, c, sigma0, boundary_cond):
    p_data = []    

    for i, x0 in enumerate(x0_sources):
        input_data = xt_grids[i]

        x_data_i = input_data[0].flatten()
        t_data_i = input_data[1].flatten()

        p_sol = WE1D(x_data_i,t_data_i,x0,boundary_cond,c,sigma0)        
        p_data.append(np.asarray(p_sol).reshape(-1,1))

    return np.asarray(p_data)