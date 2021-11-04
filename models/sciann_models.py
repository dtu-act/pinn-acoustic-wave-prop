# ==============================================================================
# Copyright 2021 Technical University of Denmark
# Author: Nikolas Borrel-Jensen 
#
# All Rights Reserved.
#
# Licensed under the MIT License.
# ==============================================================================
import sciann as sn
import os
import tensorflow as tf
from pathlib import Path
from datahandlers.meshing import TargetIndexes
from datahandlers.sciann_datagenerator import DataGeneratorXT

from models.datastructures import ADENeuralNetwork, BoundaryCondition, BoundaryType, LossType, PressureNeuralNetwork, SourceType, SciannFunctionals, Accumulators
from models.kernel_initializers import SineInitializer, first_layer_sine_init
from setup.settings import Settings

def setupPinnModels(settings: Settings, funcs: SciannFunctionals, accs: Accumulators, 
    loss_type=LossType.PINN, boundary_cond_override: BoundaryCondition=None, plot_to_file: str=None):
    """ Setup the model for pinn """

    x,t,x0,p = funcs.x, funcs.t, funcs.x0, funcs.p

    p_nn_weights = settings.network.p_nn.weights
    c = settings.physics.c
    rho = settings.physics.rho

    if boundary_cond_override == None:
        # use boundary condition from main settings
        boundary_condition = settings.domain.boundary_cond
    else:
        # use specific boundary condition (e.g. when doing transfer learning)
        boundary_condition = boundary_cond_override

     # BOUNDARY LOSSES
    if boundary_condition.type == BoundaryType.DIRICHLET:
        BC = dirichletBCLosses(p,boundary_condition,p_nn_weights)
    elif boundary_condition.type == BoundaryType.NEUMANN:        
        BC_left,BC_right = neumannBCLosses(x,p,boundary_condition,p_nn_weights)
    elif boundary_condition.type == BoundaryType.IMPEDANCE_FREQ_INDEP:
        BC_left,BC_right = freqIndependentBCLosses(x,t,p,c,boundary_condition,p_nn_weights)
    elif boundary_condition.type == BoundaryType.IMPEDANCE_FREQ_DEP:
        ade_nn_weights = settings.network.ade_nn.weights
        acc_norm = settings.network.ade_nn.accumulator_norm

        BC_left,BC_right = freqDependentBCLosses(x,t,p,rho,accs,boundary_condition,p_nn_weights,acc_norm)

        phi_acc, psi0_acc, psi1_acc = accs.phi, accs.psi0, accs.psi1        
        ODE_phi_0, ODE_phi_1, ODE_psi0_0, ODE_psi1_0 = odeLoss(t,p,boundary_condition,ade_nn_weights,acc_norm,phi_acc,psi0_acc,psi1_acc,'lr')
    else:
        raise NotImplementedError()

    if settings.domain.source.type != SourceType.IC:
        raise NotImplementedError()
    
    PDE_p = pdeLoss(x,t,p,c,p_nn_weights)
    IC,IC_t = icLosses(x,t,x0,p,settings.domain.source.source,p_nn_weights)

    if boundary_condition.type == BoundaryType.IMPEDANCE_FREQ_DEP:
        targets = [PDE_p, 
            BC_left,BC_right,
            ODE_phi_0, ODE_phi_1, ODE_psi0_0, ODE_psi1_0,
            IC_t, IC]
    elif boundary_condition.type == BoundaryType.DIRICHLET:
        targets = [PDE_p, BC, IC_t, IC]
    else:
        targets = [PDE_p, BC_left, BC_right, IC_t, IC]        

    if loss_type != LossType.PINN:
        raise NotImplementedError()    
    
    if plot_to_file != None:
        path_dir = os.path.split(plot_to_file)[0]
        Path(path_dir).mkdir(parents=True, exist_ok=True)

    m = sn.SciModel(
            inputs = [x, t, x0],
            targets = targets,
            loss_func = "mse",
            optimizer = settings.network.optimizer,
            plot_to_file=plot_to_file)

    return m

def setupDataModel(settings: Settings):
    """ Setup the model for data (only) """

    x,t,x0,p = setupNN_PDE(settings.network)
    d = sn.Data(sn.rename(p, name='data'))

    m = sn.SciModel(
        inputs = [x,t,x0],
        targets = d,
        loss_func = "mse",
        optimizer = settings.optimizer)
            
    functionals = SciannFunctionals(x,t,x0,p)

    return m,functionals

def setupPinnTargetsTrain(data: DataGeneratorXT, target_indxs: TargetIndexes, boundary_cond: BoundaryCondition):
    """ Returning the target_data for use in the training for PINN (no data).
        The order is [DOMAIN, BC_LEFT, BC_RIGHT, IC_t, IC]

        NOTE: the order should match with the x_train, t_train and x0_train vectors
              given as input in the training pass
    """

    DOMAIN_ENUM = target_indxs.domain
    IC_ENUM = target_indxs.ic
    BC_LEFT_ENUM = target_indxs.bc_left
    BC_RIGHT_ENUM = target_indxs.bc_right
    BCs = target_indxs.bc
    #ALL = target_indxs.all
    tdata = data.targets_data # ["domain", "ic", "bc-left", "bc-right", "bc", "point-source", "all"]

    if boundary_cond.type == BoundaryType.IMPEDANCE_FREQ_DEP:
        return [tdata[DOMAIN_ENUM], 
            tdata[BC_LEFT_ENUM], tdata[BC_RIGHT_ENUM],
            tdata[BCs],tdata[BCs],tdata[BCs],tdata[BCs], # aux differential equations
            tdata[IC_ENUM],tdata[IC_ENUM]] # [IC_t, IC]
    elif boundary_cond.type == BoundaryType.DIRICHLET:
        return [tdata[DOMAIN_ENUM], tdata[BCs],
            tdata[IC_ENUM],tdata[IC_ENUM]] # [IC_t, IC]
    else:        
        return [tdata[DOMAIN_ENUM],
            tdata[BC_LEFT_ENUM], tdata[BC_RIGHT_ENUM],
            tdata[IC_ENUM],tdata[IC_ENUM]] # [IC_t, IC]

def setupNN_PDE(nn: PressureNeuralNetwork):
    dtype='float64'
    x = sn.Variable("x", dtype=dtype)
    t = sn.Variable("t", dtype=dtype)
    x0 = sn.Variable("x0", dtype=dtype)

    if nn.activation=='sin':
        p = sn.Functional("p", [x, t, x0], nn.num_layers*[nn.num_neurons], nn.activation, kernel_initializer=SineInitializer)
        first_layer_sine_init(p,dtype=dtype)
    else:
        p = sn.Functional("p", [x, t, x0], nn.num_layers*[nn.num_neurons], nn.activation)

    return SciannFunctionals(x,t,x0,p,None)

def setAccumulatorsTraineable(acc: Accumulators, traineable: bool):
    if acc == None:
        return
        
    acc.phi[0].set_trainable(traineable)
    acc.phi[1].set_trainable(traineable)
    acc.psi0[0].set_trainable(traineable)
    acc.psi1[0].set_trainable(traineable)

def setupNN_ODE(funcs: SciannFunctionals, nn: ADENeuralNetwork):

    if nn.activation.lower() == "relu" or nn.activation.lower() == "elu":
        initializer = tf.keras.initializers.HeNormal()
    elif nn.activation.lower() == "tanh":
        initializer = tf.keras.initializers.GlorotNormal()
    else:
        initializer = None

    phi_0,phi_1,psi0_0,psi1_0 = sn.Functional([
        "phi_0","phi_1","psi0_0","psi1_0"],
        [funcs.x,funcs.t,funcs.x0], nn.num_layers*[nn.num_neurons], nn.activation, kernel_initializer=initializer)

    phi_acc, psi0_acc, psi1_acc = [phi_0,phi_1], [psi0_0], [psi1_0]
    
    return Accumulators(phi_acc, psi0_acc, psi1_acc)

def pdeLoss(x,t,p,c,weights):
    """ Returning PDE loss """
    w_pde = weights.pde

    p_tt = sn.math.diff(p, t, order=2)
    p_xx = sn.math.diff(p, x, order=2)
    PDE = w_pde*(p_tt - c**2 * p_xx)
    PDE = sn.rename(PDE, 'PDE')

    return PDE

def odeLoss(t,p,boundary_cond,weights,acc_factors,phi_acc,psi0_acc,psi1_acc,tag_location):
    ws_ade = weights.ade

    impedance_data = boundary_cond.impedance_data

    lambdas = impedance_data.lambdas
    alpha = impedance_data.alpha
    beta  = impedance_data.beta

    phi_norm_0,phi_norm_1,psi_norm0_0,psi_norm1_0 = acc_factors[0],acc_factors[1],acc_factors[2],acc_factors[3]

    phi_0 = phi_acc[0]
    phi_1 = phi_acc[1]
    psi0_0 = psi0_acc[0]
    psi1_0 = psi1_acc[0]

    phi_0_t = sn.math.diff(phi_0, t, order=1)
    phi_1_t = sn.math.diff(phi_1, t, order=1)
    psi0_0_t = sn.math.diff(psi0_0, t, order=1)
    psi1_0_t = sn.math.diff(psi1_0, t, order=1)
    
    ODE_phi_0  = phi_0_t + lambdas[0]*phi_0 - phi_norm_0*p
    ODE_phi_1  = phi_1_t + lambdas[1]*phi_1 - phi_norm_1*p
    ODE_psi0_0 = psi0_0_t + alpha[0]*psi0_0 + psi_norm0_0*(1/psi_norm1_0)*beta[0]*psi1_0 - psi_norm0_0*p
    ODE_psi1_0 = psi1_0_t + alpha[0]*psi1_0 - psi_norm1_0*(1/psi_norm0_0)*beta[0]*psi0_0

    # the weights are adjusted w.r.t. the normalization factor (which also impacts the relative loss)
    w0 = ws_ade[0]*(1/phi_norm_0)
    w1 = ws_ade[1]*(1/phi_norm_1)
    w0_0 = ws_ade[2]*(1/psi_norm0_0)
    w0_1 = ws_ade[3]*(1/psi_norm1_0)

    ODE_phi_0 = sn.rename(w0*ODE_phi_0, f'ODE_phi_0_{tag_location}')
    ODE_phi_1 = sn.rename(w1*ODE_phi_1, f'ODE_phi_1_{tag_location}')
    ODE_psi0_0 = sn.rename(w0_0*ODE_psi0_0, f'ODE_psi0_0_{tag_location}')
    ODE_psi1_0 = sn.rename(w0_1*ODE_psi1_0, f'ODE_psi1_0_{tag_location}')

    return ODE_phi_0, ODE_phi_1, ODE_psi0_0, ODE_psi1_0

def dirichletBCLosses(p,boundary_cond,weights):
    w_bc = weights.bc

    BC = w_bc*(p - boundary_cond.p)
    BC = sn.rename(BC, 'dirichlet_lr')

    return BC

def neumannBCLosses(x,p,boundary_cond,weights):
    w_bc = weights.bc

    p_x = sn.math.diff(p, x, order=1)
    BC_left  = w_bc*(-1*p_x - boundary_cond.v)
    BC_right = w_bc*( 1*p_x - boundary_cond.v)
    BC_left = sn.rename(BC_left, 'neumann_l')
    BC_right = sn.rename(BC_right, 'neumann_r')

    return BC_left,BC_right

def freqIndependentBCLosses(x,t,p,c,boundary_cond,weights):
    w_bc = weights.bc
    xi = boundary_cond.xi
    
    p_x = sn.math.diff(p, x, order=1)
    p_t = sn.math.diff(p, t, order=1)

    BC_left  = w_bc*(p_t + (-1*c*xi*p_x))
    BC_right = w_bc*(p_t + ( 1*c*xi*p_x))
    BC_left  = sn.rename(BC_left, 'freq_indep_l')
    BC_right = sn.rename(BC_right, 'freq_indep_r')

    return BC_left, BC_right

def freqDependentBCLosses(x,t,p,rho,acc,boundary_cond,weights,acc_factors):    
    w_bc = weights.bc
    impedance_data = boundary_cond.impedance_data

    Yinf = impedance_data.Yinf
    A = impedance_data.A
    B = impedance_data.B
    C = impedance_data.C

    phi_acc, psi0_acc, psi1_acc = acc.phi, acc.psi0, acc.psi1

    phi_denorm_0,phi_denorm_1,psi_denorm0_0,psi_denorm1_0 = 1/acc_factors[0],1/acc_factors[1],1/acc_factors[2],1/acc_factors[3]

    # normal velocity at the boundary
    vn = Yinf*p + phi_denorm_0*A[0]*phi_acc[0] + phi_denorm_1*A[1]*phi_acc[1] + 2*(psi_denorm0_0*B[0]*psi0_acc[0] + psi_denorm1_0*C[0]*psi1_acc[0])

    p_x = sn.math.diff(p, x, order=1)
    vn_t = sn.math.diff(vn, t, order=1)

    BC_left  = w_bc*(-1*p_x + (rho*vn_t))
    BC_right = w_bc*( 1*p_x + (rho*vn_t))
    BC_left = sn.rename(BC_left, 'freq_dep_l')
    BC_right = sn.rename(BC_right, 'freq_dep_r')

    return BC_left, BC_right

def icLosses(x,t,x0,p,source_f,weights):
    """ Returning IC losses: IC,IC_t """
    w_ic = weights.ic

    # Loss function, initial condition IC
    p_t0 = source_f(x,x0)
    IC = sn.rename(w_ic*(p - p_t0), 'IC')

    # Loss function, initial condition IC_t, p_t=0
    p_t = sn.math.diff(p, t, order=1)
    IC_t = sn.rename(w_ic*p_t, 'IC_t')

    return IC,IC_t

def tensorboard_callback(models_dir, model_type):
    logdir = os.path.join(models_dir, f"{model_type}/logs")
    return tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

def checkpoint_callback(models_dir, model_type):
    checkpoint_path = os.path.join(models_dir, f"{model_type}/cp.ckpt")
    checkpoint_dir = os.path.dirname(checkpoint_path)
    Path(os.path.dirname(checkpoint_dir)).mkdir(parents=True, exist_ok=True)

    # Create a callback that saves the model's weights
    return tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=False)