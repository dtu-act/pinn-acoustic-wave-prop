
# ==============================================================================
# Copyright 2021 Technical University of Denmark
# Author: Nikolas Borrel-Jensen 
#
# All Rights Reserved.
#
# Licensed under the MIT License.
# ==============================================================================
from dataclasses import dataclass
from models.datastructures import ADENeuralNetwork, BoundaryType, Domain, InputOutputDirs, NetworkSettings, Physics, PressureNeuralNetwork, TransferLearning
import setup.parsers as parsers

@dataclass
class Settings:
    verbose_out: bool
    show_plots: bool
    do_transfer_learning: bool

    dirs: InputOutputDirs
    domain: Domain    
    network: NetworkSettings
    physics: Physics

    def __init__(self, settings_dict, base_dir=None):
        self.verbose_out = settings_dict['verbose_out']
        self.show_plots = settings_dict['show_plots']
        self.do_transfer_learning = 'do_transfer_learning' in settings_dict and settings_dict['do_transfer_learning']

        if self.do_transfer_learning:
            tl_dict = settings_dict["transfer_learning"]
            
            assert tl_dict['boundary_type'] == settings_dict['boundary_type'], "Not implemented: loaded model for transfer learning should have same boundary condition as output model"

            boundary_cond = parsers.setupBoundaryCondition(settings_dict)
            self.transfer_learning = TransferLearning(boundary_cond, tl_dict["model_dir"], tl_dict["trainable"])

        boundary_cond = parsers.setupBoundaryCondition(settings_dict)
    
        Xmin = settings_dict['Xmin']
        Xmax = settings_dict['Xmax']
        tmax = settings_dict['tmax'] # input to NN should be normalized?
        ppw = settings_dict['ppw'] # points per wavelength
        x0_sources = settings_dict['source_pos']
        c = settings_dict['c']
        c_phys = settings_dict['c_phys']
        fmax = settings_dict['fmax']
        sigma0 = settings_dict['sigma0']
        rho = settings_dict['rho']        

        self.physics = Physics(sigma0,fmax,c,c_phys,rho)

        ic_points_p = settings_dict['ic_points_distr'] # percentage of points refined at t=0 (only applied for IC energy)
        bc_points_p = settings_dict['bc_points_distr'] # precentage of points refined at boundaries
        
        self.dirs = InputOutputDirs(settings_dict,base_dir=base_dir)

        lambda_w = c/fmax
        dx = lambda_w/ppw
        dt = dx/c # CFL condition (FDTD)

        self.domain = Domain([Xmin, Xmax], tmax, ppw, dt, dx, boundary_cond, sigma0, x0_sources, ic_points_p, bc_points_p)
        
        activation = settings_dict['activation']
        num_layers = settings_dict['num_layers']
        num_neurons = settings_dict['num_neurons']
        weights = parsers.setupLossPenalties(settings_dict)
        
        p_nn = PressureNeuralNetwork(activation,num_layers,num_neurons,weights)

        if boundary_cond.type == BoundaryType.IMPEDANCE_FREQ_DEP:
            activation_ade = settings_dict['activation_ade']
            num_layers_ade = settings_dict['num_layers_ade']
            num_neurons_ade = settings_dict['num_neurons_ade']
            accumulator_norm = settings_dict['accumulator_factors']
            weights: parsers.setupLossPenalties(settings_dict)
            ade_nn = ADENeuralNetwork(activation_ade,num_layers_ade,num_neurons_ade,accumulator_norm,weights)
        else:
            ade_nn = None

        epochs = settings_dict['epochs']
        stop_loss_value = settings_dict['stop_loss_value'] if 'stop_loss_value' in settings_dict else 1e-8

        batch_size = settings_dict['batch_size']
        learning_rate = settings_dict['learning_rate']
        optimizer = settings_dict['optimizer']

        self.network = NetworkSettings(epochs,stop_loss_value,batch_size,learning_rate,optimizer,p_nn,ade_nn)
