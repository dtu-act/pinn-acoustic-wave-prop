# ==============================================================================
# Copyright 2021 Technical University of Denmark
# Author: Nikolas Borrel-Jensen 
#
# All Rights Reserved.
#
# Licensed under the MIT License.
# ==============================================================================
import matplotlib.pyplot as plt
import numpy as np
import os
import datahandlers.training_data_setup as training
import datahandlers.sciann_multidatagenerator as mdg
from setup.settings import Settings
import utils.plotting as plot
import utils.animations as anim
from utils.utils import extractSignal

def evaluatePlotWaveSideBySide(m_pinn,funcs,settings,tag=''):
    """ Plot predicted wave propagation and L1 error side-by-side for each source position """

    def eval(m, funcs, x_data, t_data, p_data, x0_data, tag='', c_denorm=1, figs_dir=None):        
        p = funcs.p

        for i, _ in enumerate(x0_data):
            x_data_i = x_data[i]
            t_data_i = t_data[i]
            p_ref_data_i = p_data[i]
            x0_data_i = x0_data[i]

            p_pred_data_i = p.eval(m, [x_data_i, t_data_i, x0_data_i])

            N = len(p_pred_data_i)
            err_L1 = np.abs(p_pred_data_i - p_ref_data_i).flatten()
            mean_err = np.round(np.sum(err_L1)/N,4)

            # all x0 values are the same for each source
            tag_p_i = "p prediction (%s), x0=%0.2f" % (tag, x0_data_i[0])
            
            plot.plotSideBySide(x_data_i.flatten(), (t_data_i.flatten()/c_denorm), p_pred_data_i.flatten(), p_ref_data_i.flatten(),
                figs_dir=figs_dir, tag=tag_p_i, err=mean_err)
    
    tmax = settings.domain.tmax
    data_path = settings.dirs.data_path
    figs_dir = settings.dirs.figs_dir
    show_plots = settings.show_plots

    xt_grid,p_ref_data,_,x0_sources,_,_ = training.loadDataFromH5(data_path, tmax=tmax)

    data = mdg.MultiDataContainerXT(xt_grid)
    x0_ref_data = np.asarray([[[x0],]*len(data[i][0][0]) for i,x0 in enumerate(x0_sources)])

    # separate data
    x_ref_data = list([data[i][0][0] for i,_ in enumerate(x0_sources)])
    t_ref_data = list([data[i][0][1] for i,_ in enumerate(x0_sources)])

    eval(m_pinn, funcs, x_ref_data, t_ref_data, p_ref_data, x0_ref_data, 
        figs_dir=figs_dir, tag=tag)

    plot.plotReference(data_path,tmax,figs_dir=figs_dir)
    
    if show_plots:
        plt.show()

def evaluatePlotAccumulators(m_pinn,funcs,accum,settings:Settings,tag='',do_animations=False):
    """ Plot and animate accumulators for for each source position
    """

    def eval(m, funcs, accum, xbounds, t_data, x_data, x0_data, p_ref_data, acc_ref_l_scrs, acc_ref_r_scrs, 
        accumulator_norm, c_phys, figs_dir=None, tag='', do_animations=False):        
        
        p = funcs.p

        for j, _ in enumerate(x0_data):
            t_unique = np.unique(t_data[j])
            x_unique = np.unique(x_data[j])

            t_data_i = t_data[j]
            x_data_i = x_data[j]
            p_ref_data_i = p_ref_data[j]
            x0_data_i = np.asarray([x0_data[j][0]]*len(t_data_i)) # repeat source position corresponding to t resolution

            x_l = np.asarray([xbounds[0]]*len(t_unique)) # repeat left boundary position corresponding to t resolution
            x_r = np.asarray([xbounds[1]]*len(t_unique)) # repeat right boundary position corresponding to t resolution
            x0_unique = np.asarray([x0_data[j][0]]*len(t_unique)) # repeat source position corresponding to t resolution

            acc_ref_l = acc_ref_l_scrs[j]
            acc_ref_r = acc_ref_r_scrs[j]
            
            acc_pred_l = []
            acc_pred_r = []

            labels_acc = ['$\phi_0$', '$\phi_1$', '$\psi^{(0)}_0$', '$\psi^{(1)}_0$']
            acc_phi = [accum.phi[0], accum.phi[1]]
            acc_psi = [accum.psi0[0], accum.psi1[0]]

            for i,_ in enumerate(acc_phi):
                acc_phi_i = acc_phi[i]

                acc_pred_l.append(acc_phi_i.eval(m, [x_l,t_unique,x0_unique])/accumulator_norm[i])
                acc_pred_r.append(acc_phi_i.eval(m, [x_r,t_unique,x0_unique])/accumulator_norm[i])

            for i,_ in enumerate(acc_psi):
                acc_psi_i = acc_psi[i]

                acc_pred_l.append(acc_psi_i.eval(m, [x_l,t_unique,x0_unique])/accumulator_norm[2+i])
                acc_pred_r.append(acc_psi_i.eval(m, [x_r,t_unique,x0_unique])/accumulator_norm[2+i])

            p_pred_data_i = p.eval(m, [x_data_i, t_data_i, x0_data_i])
            p_pred = p_pred_data_i.reshape(len(t_unique),len(x_unique))
            p_ref = p_ref_data_i.reshape(len(t_unique),len(x_unique))

            x0 = x0_data[j].flatten()[0]

            plot.plotAccumulators(t_unique/c_phys, acc_pred_l, acc_pred_r, acc_ref_l, acc_ref_r, x0, labels_acc, figs_dir=figs_dir, tag=f'{tag}_src{j}')

            if do_animations and i == 0:
                anim.animateAccumulators(t_unique/c_phys, x_unique, p_pred, p_ref, 
                    acc_pred_l, acc_pred_r, acc_ref_l, acc_ref_r, f'Accumulators at $x_0={x0}$', 
                    labels_acc, figs_dir=figs_dir, tag=f'{tag}_src{j}')

    data_path = settings.dirs.data_path
    tmax = settings.domain.tmax
    accumulator_norm = settings.network.ade_nn.accumulator_norm
    figs_dir = settings.dirs.figs_dir
    show_plots=False
    do_animations=False

    xt_grid,p_data,_,x0_sources,acc_ref_l_srcs,acc_ref_r_srcs = training.loadDataFromH5(data_path, tmax=tmax)

    data = mdg.MultiDataContainerXT(xt_grid)
    x0_data = np.asarray([[[x0],]*len(data[i][0][0]) for i,x0 in enumerate(x0_sources)])

    # separate data
    x_data = list([data[i][0][0] for i,_ in enumerate(x0_sources)])
    t_data = list([data[i][0][1] for i,_ in enumerate(x0_sources)])

    xbounds = [np.min(data[0][0][0]), np.max(data[0][0][0])]

    eval(m_pinn, funcs, accum, xbounds, t_data, x_data, x0_data, p_data, acc_ref_l_srcs,acc_ref_r_srcs, 
        accumulator_norm, settings.physics.c_phys, figs_dir=figs_dir, tag=tag, do_animations=do_animations)
    
    if show_plots:
        plt.show()

def evaluatePlotIR_TF(m,funcs,settings,r0_list,c_phys,figs_dir=None):
    """ Evaluate the surrogate model and plot impulse response 
        and transfer function for each source position
    """

    def calcErrors(p_pred_data, p_ref_data, x0, r0, label=''):
        # find indexes for value diff. greater than -60dB
        threshold_dB = 60
        indxs_ref = np.where(20*np.log(p_ref_data/np.max(p_ref_data)) > -threshold_dB)[0]
        indxs_pred = np.where(20*np.log(p_pred_data/np.max(p_ref_data)) > -threshold_dB)[0]
        indxs = np.union1d(indxs_ref, indxs_pred)

        err_L1 = np.abs(p_pred_data - p_ref_data)
        err_rel = err_L1[indxs]/p_ref_data[indxs]
        mean_err = np.round(np.mean(err_L1),4)
        mean_err_rel = np.round(np.mean(err_rel),4)

        print('---------------')
        print(f'Domain: {label}')
        print(f'(src,rec) = ({x0},{np.round(r0,3)})')
        print(f'Mean/max err: {mean_err} / {np.round(max(err_L1),3)}')
        print(f'Mean rel. err: {np.round(mean_err_rel*100,1)}% / {np.round(20*np.log10(1 - mean_err_rel), 1)} dB')
        #print(f'Max rel. err %/dB: {np.round(max(err_rel),3)} / {20*np.log(err_rel)}')
        print('---------------')

        return mean_err, mean_err_rel, err_L1, err_rel

    data_path = settings.dirs.data_path
    tmax = settings.domain.tmax
    if figs_dir==None:
        figs_dir = settings.dirs.figs_dir

    xt_grid,p_ref_data,_,x0_sources,_,_ = training.loadDataFromH5(data_path, tmax=tmax)

    data = mdg.MultiDataContainerXT(xt_grid)
    x0_data = np.asarray([[[x0],]*len(data[i][0][0]) for i,x0 in enumerate(x0_sources)])

    # separate data
    x_data = list([data[i][0][0] for i,_ in enumerate(x0_sources)])
    t_data = list([data[i][0][1] for i,_ in enumerate(x0_sources)])

    p = funcs.p
        
    for i, x0 in enumerate(x0_data):
        x_data_i = x_data[i]
        t_data_i = t_data[i]
        p_ref_data_i = p_ref_data[i]
        x0_data_i = x0_data[i]
        r0 = r0_list[i]

        p_pred_data_i = p.eval(m, [x_data_i, t_data_i, x0_data_i])

        p_r0_ref,t_ref = extractSignal(r0,x_data_i,t_data_i,p_ref_data_i)
        p_r0_pred,t_pred = extractSignal(r0,x_data_i,t_data_i,p_pred_data_i)
        
        path_file_td = os.path.join(figs_dir, "td_x0=%0.2f.png" % x0_data_i[0])
        path_file_tf = os.path.join(figs_dir, "tf_x0=%0.2f.png" % x0_data_i[0])        

        plot.plotTimeDomain(p_r0_ref,p_r0_pred,t_ref/c_phys, t_pred/c_phys, show_legends=i==0, path_file=path_file_td)
        plot.plotTransferFunction(p_r0_pred, p_r0_ref, tmax/c_phys, freq_min_max=[20, 1000], show_legends=i==0, path_file=path_file_tf)

        calcErrors(p_r0_pred, p_r0_ref, x0[0], r0, label='Time domain')

def evaluatePlotAtReceiverPositions(m,funcs,settings,r0_list,figs_dir=None):
    """ Evaluate the surrogate model and plot the prediction, reference and L1 error for each source position
    """
    
    data_path = settings.dirs.data_path
    tmax = settings.domain.tmax
    
    if figs_dir==None:
        figs_dir = settings.dirs.figs_dir

    xt_grid,p_ref_data,_,x0_sources,_,_ = training.loadDataFromH5(data_path, tmax=tmax)

    data = mdg.MultiDataContainerXT(xt_grid)
    x0_data = np.asarray([[[x0],]*len(data[i][0][0]) for i,x0 in enumerate(x0_sources)])

    # separate data
    x_data = list([data[i][0][0] for i,_ in enumerate(x0_sources)])
    t_data = list([data[i][0][1] for i,_ in enumerate(x0_sources)])

    p = funcs.p

    for i, _ in enumerate(x0_data):    
        x_data_i = x_data[i]
        t_data_i = t_data[i]
        p_ref_data_i = p_ref_data[i]
        x0_data_i = x0_data[i]
        x0 = x0_data_i[0]
        r0 = r0_list[i]

        p_pred_data_i = p.eval(m, [x_data_i, t_data_i, x0_data_i])
        
        err_L1 = np.abs(p_pred_data_i - p_ref_data_i).flatten()

        # all x0 values are the same for each source - extract first [0]
        path_file_ref = os.path.join(figs_dir, "p_ref_x0=%0.2f.png" % x0)
        path_file_pred = os.path.join(figs_dir, "p_pred_x0=%0.2f.png" % x0)
        path_file_err = os.path.join(figs_dir, "err_L1_x0=%0.2f.png" % x0)
        
        path_cbar_file_ref = os.path.join(figs_dir, "cbar_ref_x0=%0.2f.png" % x0)
        path_cbar_file_pred = os.path.join(figs_dir, "cbar_pred_x0=%0.2f.png" % x0)
        path_cbar_file_err = os.path.join(figs_dir, "cbar_err_L1_x0=%0.2f.png" % x0)
        
        label_str = 'Receiver' if i==0 else None

        plot.plotData(x_data_i.flatten(), t_data_i.flatten(), p_ref_data_i.flatten(), 
            v_minmax=[0,1], vline=[r0,'--', 4, 'red', label_str],
            path_file=path_file_ref, path_cbar_file=path_cbar_file_ref)
        plot.plotData(x_data_i.flatten(), t_data_i.flatten(), p_pred_data_i.flatten(), 
            v_minmax=[0,1], vline=[r0,'-', 4, 'blue', label_str],
            path_file=path_file_pred, path_cbar_file=path_cbar_file_pred)
        plot.plotData(x_data_i.flatten(), t_data_i.flatten(), err_L1, 
            v_minmax=[0,0.015], vline=[r0,'-', 4, 'orange', label_str],
            path_file=path_file_err, path_cbar_file=path_cbar_file_err)

def evaluateAnimateWave(m, funcs, settings, receiver_pos, c_phys, title=''):
    """ Animate wave propagation for each source position """
    data_path = settings.dirs.data_path
    tmax = settings.domain.tmax
    figs_dir = settings.dirs.figs_dir

    xt_grid,p_ref_data,_,x0_sources,_,_ = training.loadDataFromH5(data_path, tmax=tmax)

    data = mdg.MultiDataContainerXT(xt_grid)
    x0_data = np.asarray([[[x0],]*len(data[i][0][0]) for i,x0 in enumerate(x0_sources)])

    # separate data
    x_data = list([data[i][0][0] for i,_ in enumerate(x0_sources)])
    t_data = list([data[i][0][1] for i,_ in enumerate(x0_sources)])

    p = funcs.p

    p_preds = []
    p_refs = []
    p_r0_preds = []
    p_r0_refs = []

    for i, _ in enumerate(x0_data):
        x_data_i = x_data[i]
        t_data_i = t_data[i]
        p_data_i = p_ref_data[i]
        x0_data_i = x0_data[i]

        p_pred_data_i = p.eval(m, [x_data_i, t_data_i, x0_data_i])

        r0 = receiver_pos[i]

        t = np.unique(t_data_i.flatten())
        x = np.unique(x_data_i.flatten())

        p_r0_ref,_ = extractSignal(r0,x_data_i,t_data_i,p_data_i)
        p_r0_pred,_ = extractSignal(r0,x_data_i,t_data_i,p_pred_data_i)

        p_pred = p_pred_data_i.reshape(len(t),len(x))
        p_ref = p_data_i.reshape(len(t),len(x))

        p_preds.append(p_pred)
        p_refs.append(p_ref)
        p_r0_preds.append(p_r0_pred)
        p_r0_refs.append(p_r0_ref)

    anim.plotAnimations(x, t/c_phys, p_preds, p_refs, p_r0_preds, p_r0_refs, 
        x0_sources.flatten(), receiver_pos, title=title, figs_dir=figs_dir)