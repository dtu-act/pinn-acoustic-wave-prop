# ==============================================================================
# Copyright 2021 Technical University of Denmark
# Author: Nikolas Borrel-Jensen 
#
# All Rights Reserved.
#
# Licensed under the MIT License.
# ==============================================================================
import os
import h5py
from pathlib import Path
import numpy as np
from models.datastructures import Domain, Physics

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
            tmax = f.attrs['tmax'][0]

            x0_sources = f['x0_sources'][()] 
            tmax_grid = max(f['grids'][0][1].flatten()) # python list

            assert tmax == tmax_grid, "tmax is inconsistent"

            physics_settings = Physics(sigma0,fmax,c,343,rho)

        return dt,dx,x0_sources,physics_settings,tmax
        
def loadDataFromH5(path_data, tmax=None):
    """ input
            tmax: normalized max time 
        output
            grids: is a S X x X y x t dimensional array [i,j]
    """

    acc_l_data = []
    acc_r_data = []

    with h5py.File(path_data, 'r') as f:
        dt = f.attrs['dt'][0]

        x0_sources = f['x0_sources'][()]
        grids_all = np.array(f['grids'][()])
        sols = f['solutions'][()]

        if tmax == None:
            #ilast = len(t)-1
            ilast = len(grids_all[0,1,:,0]) - 1
            grids = grids_all.tolist()
        else:
            #ilist = [i for i, n in enumerate(t) if abs(n - tmax) < dt/2]
            # assume all grids have same number of element (analyze first grids[0])
            ilist = [i for i, n in enumerate(grids_all[0,1,:,0]) if abs(n - tmax) <= dt/2]
            if not ilist:
                raise Exception('t_max exceeds simulation data running time')
            ilast = ilist[0]
            grids = grids_all[:,:,:ilast+1,:].tolist() # crop w.r.t. time

        for i,_ in enumerate(x0_sources):
            if f'acc_l{i}' in f.keys() and f'acc_r{i}' in f.keys():
                acc_l = f[f'acc_l{i}'][:,:ilast+1]
                acc_r = f[f'acc_r{i}'][:,:ilast+1]
                acc_l_data.append(acc_l)
                acc_r_data.append(acc_r)

    return grids,sols,x0_sources,acc_l_data,acc_r_data

def writeDataToHDF5(grids, solutions, domain: Domain, physics: Physics, path_file: str):
    """ Write data as HDF5 format
    """
    """ Ex: writeHDF5([1,2,3],[0.1,0.2,0.3],data,[-0.2,0.0,0.2],0.1,1.0,343,0.1,2000,'test1.h5')
    """
    x0_sources = domain.x0_sources
    dt = domain.dt
    dx = domain.dx
    tmax = domain.tmax

    c = physics.c
    rho = physics.rho
    sigma0 = physics.sigma0
    fmax = physics.fmax    

    assert(len(solutions) == len(x0_sources))
    
    if Path(path_file).exists():
        os.remove(path_file)

    with h5py.File(path_file, 'w') as f:
        f.create_dataset('x0_sources', data=x0_sources)
        f.create_dataset('grids', data=grids)
        f.create_dataset('solutions', data=solutions)
        
        # wrap in array to be consistent with data generated from Matlab
        f.attrs['dt'] = [dt]
        f.attrs['dx'] = [dx]
        f.attrs['c'] = [c]
        f.attrs['rho'] = [rho]
        f.attrs['sigma0'] = [sigma0]
        f.attrs['fmax'] = [fmax]
        f.attrs['tmax'] = [tmax]
        
        print(list(f.keys()))
        print(list(f.attrs))