# ==============================================================================
# Copyright 2021 Technical University of Denmark
# Author: Nikolas Borrel-Jensen 
#
# All Rights Reserved.
#
# Licensed under the MIT License.
# ==============================================================================
#import sys; sys.path.append('path/to/sciann') # for modified github code
import tensorflow as tf
import numpy as np
import os, shutil
import matplotlib.pyplot as plt

from setup.settings import Settings
import utils.plotting as plot
import setup.configurations as configs
import setup.parsers as parsers
import utils.evaluate as ueval
import models.sciann_models as models
from models.datastructures import LossType, BoundaryType
import datahandlers.meshing as meshing
from utils.validations import printGridStats, printSettings, validateData, plotGridSource

def train(settings_path):
    configs.setupPlotParams()
    #tf.debugging.set_log_device_placement(True)
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        
    settings_dict = parsers.parseSettings(settings_path)
    settings = Settings(settings_dict)
    
    settings.dirs.createDirs(delete_existing=True)
    shutil.copyfile(settings_path, os.path.join(settings.dirs.id_dir, 'settings.json')) # copy settings
    validateData(settings)

    tb_callback_f = lambda model_type: models.tensorboard_callback(settings.dirs.models_dir, model_type)
    cp_callback_f = lambda model_type: models.checkpoint_callback(settings.dirs.models_dir, model_type)
    
    printSettings(settings_path)
    printGridStats(settings.domain)    
    
    ### SETUP INPUT MESH DATA ###
    data, x0_input_data, target_indxs = meshing.generateNonUniformMesh(settings.domain)
    
    plotGridSource(data, settings)
    
    t_pinn_train = np.asarray(data.inputs_data[1])
    x0_pinn_train = np.asarray(x0_input_data).reshape(-1,1)

    if settings.domain.spatial_dimension == 1:
        x_pinn_train = np.asarray(data.inputs_data[0])
        input_data = [x_pinn_train, t_pinn_train, x0_pinn_train]
    elif settings.domain.spatial_dimension == 2:
        x_pinn_train = np.asarray(data.inputs_data[0])
        y_pinn_train = np.asarray(data.inputs_data[0])
        input_data = [x_pinn_train, y_pinn_train, t_pinn_train, x0_pinn_train]
    else:
        raise NotImplementedError()

    ### TRAIN AND EVALUATE ###
    funcs = models.setupNN_PDE(settings)
    accs = models.setupNN_ODE(funcs, settings.network.ade_nn) if settings.domain.boundary_cond.type == BoundaryType.IMPEDANCE_FREQ_DEP else None

    if settings.do_transfer_learning:
        m_pinn, targets_pinn = models.loadModel(settings, funcs, accs, data, target_indxs)

        # NOT IMPLEMENTED IN THIS VERSION: transfer weights from initial simple (e.g. NEUMANN/DIRECHLET) model to full impedance model
        #m_pinn = models.setupPinnModels(settings, funcs, accs, plot_to_file=settings.dirs.plot_graph_path)
        #targets_pinn = models.setupPinnTargetsTrain(data, target_indxs, settings.domain.boundary_cond)

        ueval.evaluatePlotWaveSideBySide(m_pinn,funcs,settings,tag='WEIGHTS')
        if settings.domain.boundary_cond.type == BoundaryType.IMPEDANCE_FREQ_DEP:
            ueval.evaluatePlotAccumulators(m_pinn,funcs,accs,settings,tag='LOADED')
    else:
        m_pinn = models.setupPinnModels(settings, funcs, accs, plot_to_file=settings.dirs.plot_graph_path)
        targets_pinn = models.setupPinnTargetsTrain(data, target_indxs, settings.domain.boundary_cond)
        
    m_pinn.summary()

    h_pinn = m_pinn.train(
        input_data, targets_pinn,
        batch_size=settings.network.batch_size, epochs=settings.network.epochs, 
        learning_rate=settings.network.learning_rate, stop_loss_value=settings.network.stop_loss_value,
        callbacks=[cp_callback_f(LossType.PINN), tb_callback_f(LossType.PINN)], verbose=settings.verbose_out)

    plot.plotConvergence(h_pinn, tag='TRAINED', figs_dir=settings.dirs.figs_dir)

    ueval.evaluatePlotWaveSideBySide(m_pinn,funcs,settings,tag='TRAINED')
    if settings.domain.boundary_cond.type == BoundaryType.IMPEDANCE_FREQ_DEP:
        ueval.evaluatePlotAccumulators(m_pinn,funcs,accs,settings,tag='TRAINED')

    if settings.show_plots:
        plt.show()