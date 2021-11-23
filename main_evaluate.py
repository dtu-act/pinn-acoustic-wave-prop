# ==============================================================================
# Copyright 2021 Technical University of Denmark
# Author: Nikolas Borrel-Jensen 
#
# All Rights Reserved.
#
# Licensed under the MIT License.
# ==============================================================================
import tensorflow as tf
import datahandlers.sciann_multidatagenerator as mdg
import os
from pathlib import Path
from setup.settings import Settings
from setup.setup_nn import setupPinnModelFromWeights

import utils.utils as utils
from utils.validations import validateData
from utils.evaluate import evaluatePlotIR_TF, evaluatePlotAtReceiverPositions, evaluatePlotWaveSideBySide, evaluatePlotAccumulators, evaluateAnimateWave
import setup.configurations as configs
import setup.parsers as parsers
from models.datastructures import BoundaryType
import datahandlers.reference_data_setup as ref

### SETTINGS ###
id_dir = 'freq_dep_sine_3_256_7sources_d01'
settings_filename = 'settings_srcs.json'
base_dir = "/Users/nikolasborrel/data/pinn"

do_plots_for_paper = False
do_animations = True
do_side_by_side_plot = True

### SETUP ###
configs.setupPlotParams(do_plots_for_paper)
settings_path = os.path.join(base_dir, f"results/{id_dir}/{settings_filename}")

settings_dict = parsers.parseSettings(settings_path)
settings = Settings(settings_dict, base_dir=base_dir)
settings.dirs.createDirs(False)
validateData(settings)

c_phys = settings.physics.c_phys

# LOAD REFERENCE GRID
xt_grid,_,_,_,_,_ = ref.loadDataFromH5(settings.dirs.data_path, tmax=settings.domain.tmax)
data = mdg.MultiDataContainer(xt_grid)
r0 = utils.calcSourcePositions(data,settings.domain.x0_sources)

checkpoint_path = os.path.join(settings.dirs.models_dir, 'LossType.PINN')
latest = tf.train.latest_checkpoint(checkpoint_path)
if latest == None:
    raise FileNotFoundError(f'Weights not found: %s', checkpoint_path)
m, funcs, accum = setupPinnModelFromWeights(latest, settings)

# PLOT
if do_plots_for_paper:
    configs.setupPlotParams(True)
    figs_paper_dir = os.path.join(settings.dirs.figs_dir, "paper"); Path(figs_paper_dir).mkdir(parents=True, exist_ok=True)
    evaluatePlotAtReceiverPositions(m,funcs,settings,r0,figs_dir=figs_paper_dir)
    evaluatePlotIR_TF(m,funcs,settings,r0,c_phys=settings.physics.c_phys,figs_dir=figs_paper_dir)

if do_animations:
    configs.setupPlotParams(False)

    if settings.domain.boundary_cond.type == BoundaryType.IMPEDANCE_FREQ_DEP:
        title = 'Boundary condition: Frequency-Dependent Impedance'
    elif settings.domain.boundary_cond.type == BoundaryType.IMPEDANCE_FREQ_INDEP:
        title = 'Boundary condition: Frequency-Independent Impedance'            
    elif settings.domain.boundary_cond.type == BoundaryType.NEUMANN:
        title = 'Boundary condition: Reflective Walls (Neumann)'
    else:
        title = ''
    evaluateAnimateWave(m, funcs, settings, r0, settings.physics.c_phys, title=title)

if do_side_by_side_plot:
    configs.setupPlotParams(False)

    evaluatePlotWaveSideBySide(m,funcs,settings,tag='TRAINED')
    if settings.domain.boundary_cond.type == BoundaryType.IMPEDANCE_FREQ_DEP:
        evaluatePlotAccumulators(m,funcs,accum,settings,tag='TRAINED',do_animations=do_animations)