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

import datahandlers.meshing as meshing
import models.sources as sources
from models.datastructures import Domain, Physics, SourceType, BoundaryType, SourceInfo
import datahandlers.reference_data_setup as ref
import setup.configurations as configs
import setup.parsers as parsers
from utils.validations import printGridStats
from utils.plotting import plotReference

### PATH TO EVALUATON SETTINGS ###
base_dir = "/Users/nikolasborrel/data/pinn"

boundary_type = "NEUMANN" # DIRICHLET | NEUMANN
#x0_sources = [[-0.3],[-0.15],[0.0],[0.15],[0.3]]
x0_sources = [[-0.3],[-0.2],[-0.1],[0.0],[0.1],[0.2],[0.3]]
Xminmax = [[-1], [1]]
#x0_sources = [[-0.3,-0.3],[-0.2,-0.2],[-0.1,-0.1],[0.0,0.0],[0.1,0.1],[0.2,0.2],[0.3,0.3]]
#Xminmax = [[-1,-1], [1,1]]
grid_type = 'uniform' #'non-uniform' | 'uniform'
tmax = 4 # normalized

configs.setupPlotParams()
boundary_cond = parsers.setupBoundaryCondition({'boundary_type': boundary_type})

# ACOUSTIC SETUP
c = 1
c_phys = 343
fmax_phys = 1000
fmax = fmax_phys/c_phys

physics = Physics(sigma0=0.2,fmax=fmax,c=c,c_phys=c_phys,rho=1.2)

ppw = 4
lambda_w = c/fmax
dx = lambda_w/ppw
dt = dx/c # CFL condition

ic_p = 0.25; bc_p = 0.45

spatial_dim = np.asarray(x0_sources).shape[1]

source = SourceInfo(SourceType.IC, physics.sigma0, sources.sciann_gaussianIC(physics.sigma0))
domain = Domain(Xminmax=Xminmax, tmax=tmax, ppw=ppw, dt=dt, dx=dx, boundary_cond=boundary_cond, 
    source=source, x0_sources=x0_sources, ic_points_p=ic_p, bc_points_p=bc_p)

# compose filename from parameters
src_tag = f'_srcs{len(x0_sources)}' if len(x0_sources) > 1 else ''
filename = f"{boundary_type.lower()}_{spatial_dim}D_{fmax_phys}Hz_sigma{physics.sigma0}_c{c}{src_tag}.hdf5"

printGridStats(domain)

if not (boundary_cond.type == BoundaryType.DIRICHLET or boundary_cond.type == BoundaryType.NEUMANN):
    raise Exception('Cannot generate reference data for given boundary conditions')

if grid_type == 'uniform':
    path_output_data = os.path.join(base_dir, f"reference_data/uniform/{filename}")
    data,x0_eval_data,target_sim_indxs = meshing.generateUniformMesh(domain)    
    grids = list(map(lambda x: x[0], data)) # extract mesh (without target) into array
    p_eval_data = ref.generateSolutionData1D(grids, x0_sources, c, physics.sigma0, boundary_cond.type)
elif grid_type == 'non-uniform':
    path_output_data = os.path.join(base_dir, f"reference_data/nonuniform/{filename}")
    data, x0_input_data, target_sim_indxs = meshing.generateNonUniformMesh(domain)
    grids = list(map(lambda x: x[0], data)) # extract mesh (without target) into array
    p_eval_data = ref.generateSolutionData1D(grids, x0_sources, c, physics.sigma0, boundary_cond.type)
    x0_eval_data = np.asarray([[x0,]*len(data[i][0][0]) for i,x0 in enumerate(x0_sources)])
else:
    raise NotImplementedError()

xdata = np.unique(np.asarray([grids[0][0]]))
tdata = np.unique(np.asarray([grids[0][1]]))

ref.writeDataToHDF5(xdata,tdata,p_eval_data,domain,physics,path_output_data)

plotReference(path_output_data, block_plot=True)