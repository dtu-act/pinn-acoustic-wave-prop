# ==============================================================================
# Copyright 2021 Technical University of Denmark
# Author: Nikolas Borrel-Jensen 
#
# All Rights Reserved.
#
# Licensed under the MIT License.
# ==============================================================================
import models.sciann_models as models
from models.datastructures import BoundaryType, LossType
from setup.settings import Settings

def setupPinnModelFromWeights(weights_path: str, settings: Settings, loss_type=LossType.PINN):
    if weights_path == None:
        raise FileNotFoundError('Weights not found on disk')
    
    funcs = models.setupNN_PDE(settings)
    accs = models.setupNN_ADE(funcs, settings.network.ade_nn) if settings.domain.boundary_cond.type == BoundaryType.IMPEDANCE_FREQ_DEP else None
    m = models.setupPinnModels(settings, funcs, accs, loss_type=loss_type)
    m.load_weights(weights_path)

    return m, funcs, accs