# ==============================================================================
# Copyright 2021 Technical University of Denmark
# Author: Nikolas Borrel-Jensen 
#
# All Rights Reserved.
#
# Licensed under the MIT License.
# ==============================================================================
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as anim
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec

import datahandlers.training_data_setup as training
import datahandlers.sciann_multidatagenerator as mdg
from utils.dsp import get_fft_values

res = 160
colormap = cm.magma_r #'Greys' #cm.cividis
figsize_x, figsize_y = 8, 4

def plotReference(training_data_path,tmax,plotnth=1,figs_dir=None):
    def subPlot(x_train, t_train, p_train, x0, fig, ax, plotnth=1):    
        p1 = ax.tricontourf(
            x_train[::plotnth], t_train[::plotnth], p_train[::plotnth], res, cmap=colormap)

        ax.set_title('x0=%0.2f' % x0)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(p1, cax=cax, orientation='vertical')

    xt_grid,p_data,_,x0_sources,_,_ = training.loadDataFromH5(training_data_path, tmax=tmax)    
    data = mdg.MultiDataContainerXT(xt_grid)

    fig, _ = plt.subplots(int(np.ceil(len(x0_sources)/2)),
                          min(len(x0_sources), 2), figsize=(12, 12))

    fig.suptitle('Solutions for source positions')

    for i, x0 in enumerate(x0_sources):
        ax = fig.axes[i]
        input_data_i, _ = data[i]

        x_source_i = input_data_i[0].flatten()
        t_source_i = input_data_i[1].flatten()
        p_source_i = p_data[i].flatten()

        subPlot(x_source_i, t_source_i, p_source_i, x0, fig, ax, plotnth=plotnth)

        if np.mod(i, 2) == 0:
            ax.set(xlabel='x [m]', ylabel='t [sec]')
        else:
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)

    if figs_dir != None:
        path_plot = os.path.join(figs_dir, 'p_exact_sources.png')
        fig.savefig(path_plot)

def plotSideBySide(x_mesh, t_mesh, p_pred, p_exact, figs_dir=None, plotnth=1, tag='', err=''):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    fig.suptitle('%s, err = %e' % (tag, err))

    p1 = ax1.tricontourf(
        x_mesh[::plotnth], t_mesh[::plotnth], p_pred[::plotnth], res, cmap=colormap)

    ax1.set_title('$Prediction, NN(x,t,s)\simeq p(x,t,x0)$')
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(p1, cax=cax, orientation='vertical')

    p2 = ax2.tricontourf(x_mesh[::plotnth], t_mesh[::plotnth], np.abs(
        p_exact[::plotnth] - p_pred[::plotnth]), res, cmap=colormap)

    ax2.set(xlabel='x [m]m')
    ax2.set_title('Error, $|p(x,t,x0)-NN(x,t,x0)|$')
    plt.setp(ax2.get_yticklabels(), visible=False)
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(p2, cax=cax, orientation='vertical')

    if figs_dir != None:
        path_plot = os.path.join(figs_dir, f'{tag}.png')
        plt.savefig(path_plot)

    plt.show(block=False)

def plotData(x_mesh, t_mesh, p, vline=None, path_file=None, path_cbar_file=None, v_minmax=[]):
    fig = plt.figure(figsize=(figsize_x, figsize_y))
    if v_minmax:
        cax = plt.tricontourf(x_mesh, t_mesh, p, res, cmap=colormap, vmin=v_minmax[0], vmax=v_minmax[1])
    else:
        cax = plt.tricontourf(x_mesh, t_mesh, p, res, cmap=colormap)

    if vline:
        plt.axvline(x=vline[0], linestyle=vline[1], linewidth=vline[2], color=vline[3], label=vline[4])
        if vline[4]:
            plt.legend()        
    
    fig.axes[0].get_xaxis().set_visible(False)
    fig.axes[0].get_yaxis().set_visible(False)
    plt.tight_layout()
    #plt.axis('off')
    #fig.axes[0].get_xaxis().set_ticks([-1.0, 0.0, 1.0])
    #fig.axes[0].get_xaxis().set_ticklabels(['      -1.0', '0.0', '1.0     '])
    #plt.xlabel('x [m]')
    #plt.ylabel('t [s]')
    # plt.title(title_str)
    # plt.colorbar()

    if path_file != None:
        plt.savefig(path_file,bbox_inches='tight',pad_inches=0)
    
    if path_cbar_file != None:
        fig,ax = plt.subplots(figsize=(4, 4))
        
        if v_minmax:
            prec = 3
            tick_low = v_minmax[0] #min(u[::plotnth])
            tick_high = v_minmax[1] #max(u[::plotnth])
            #tick_mid = (tick_high-tick_low)/2
            tick_mid = (v_minmax[1]-v_minmax[0])/2
            cbar = plt.colorbar(cax,ax=ax,ticks=[tick_low, tick_mid, tick_high]) #, orientation="horizontal"
            cbar.set_ticklabels([f'{round(tick_low,prec)}', f'{round(tick_mid,prec)}', f'>{round(tick_high,prec)}'])
        else:
            cbar = plt.colorbar(cax,ax=ax)
        
        cbar.ax.tick_params(labelsize=12)

        ax.remove()
        plt.savefig(path_cbar_file,bbox_inches='tight',pad_inches=0, dpi=800)

    plt.show(block=False)

def plotAccumulators(t, acc_pred_l, acc_pred_r, acc_ref_l, acc_ref_r, x0, labels_acc, figs_dir=None, tag=''):
    fig, _ = plt.subplots(4, 2, figsize=(15, 10))
    fig.suptitle(f'Accumulators for $x_0={x0}$', fontsize=20)

    for i, _ in enumerate(acc_pred_l):
        label_str = labels_acc[i]

        acc_pred_l_i = acc_pred_l[i]
        acc_ref_l_i  = acc_ref_l[i,:]

        acc_pred_r_i = acc_pred_r[i]
        acc_ref_r_i  = acc_ref_r[i,:]

        ax = fig.axes[i*2]
        ax.plot(t, acc_ref_l_i, label='ref', linestyle='--', color='red')
        ax.plot(t, acc_pred_l_i, label='pred', linestyle='-', color='blue')        
        ax.set(xlabel='t', ylabel=label_str)
        if i == 0:
            ax.set_title('left boundary', fontsize=15)
        ax.legend()

        ax = fig.axes[i*2+1]
        ax.plot(t, acc_ref_r_i, label='ref', linestyle='--', color='red')
        ax.plot(t, acc_pred_r_i, label='pred', linestyle='-', color='blue')
        ax.set(xlabel='t', ylabel=label_str)
        if i == 0:
            ax.set_title('right boundary', fontsize=15)
        ax.legend()

    if figs_dir != None:
        path_plot = os.path.join(figs_dir, f'accumulators_{tag}.png')
        fig.savefig(path_plot)

def plotTransferFunction(p_pred_data, p_ref_data, tmax, freq_min_max=[0,np.inf], show_legends=True, path_file=None):
        N = len(p_pred_data)
        dt = tmax/N
        fs = 1/dt

        print(f'f_s = {fs}')

        f_values_pred, fft_values_pred = get_fft_values(p_pred_data.flatten(), fs, NFFT=1024)
        f_values_ref, fft_values_ref = get_fft_values(p_ref_data.flatten(), fs, NFFT=1024)

        indx_min = np.where(freq_min_max[0] > f_values_pred)[0][-1]
        indx_max = np.where(freq_min_max[1] < f_values_pred)[0][0]

        f_values_pred = f_values_pred[indx_min:indx_max]
        fft_values_pred = fft_values_pred[indx_min:indx_max]
        f_values_ref = f_values_ref[indx_min:indx_max]
        fft_values_ref = fft_values_ref[indx_min:indx_max]

        fig = plt.figure(figsize=(figsize_x, figsize_y))
        plt.plot(f_values_ref, 20*np.log(fft_values_ref), linestyle='--', linewidth=4, color='red')
        plt.plot(f_values_pred, 20*np.log(fft_values_pred), linestyle='-', linewidth=4, color='blue')
        if show_legends:
            plt.legend(['Ref', 'Pred'])
        
        plt.grid()
        plt.ylim([-250,-50])
        
        #plt.axis('off')
        #fig.axes[0].get_xaxis().set_visible(False)
        #fig.axes[0].get_yaxis().set_visible(False)
        fig.axes[0].get_xaxis().set_ticks([20,200,600,1000])
        fig.axes[0].get_xaxis().set_ticklabels(['20','200','600','1000    '])
        plt.xlabel('Frequency [Hz]')
        
        if path_file != None:
            plt.savefig(path_file,bbox_inches='tight',pad_inches=0)

        plt.show(block=False)

def plotTimeDomain(p_pred_data, p_ref_data, t_pred_data, t_ref_data, show_legends=True, path_file=None):                
        fig = plt.figure(figsize=(figsize_x, figsize_y))
        plt.plot(t_pred_data, p_pred_data, linestyle='--', linewidth=4, color='red')
        plt.plot(t_ref_data, p_ref_data, linestyle='-', linewidth=4, color='blue')
        if show_legends:
            plt.legend(['Ref', 'Pred'])
        
        #fig.axes[0].get_yaxis().set_visible(False)
        plt.grid()        
        plt.ylim([0.0,0.5])
        fig.axes[0].get_yaxis().set_ticks([-0.005, 0.125, 0.25, 0.375, 0.505])
        fig.axes[0].get_yaxis().set_ticklabels(['0', '0.125', '0.25', '0.375', '0.5'])
        #plt.xlabel('t [sec]')
        
        if path_file != None:
            plt.savefig(path_file,bbox_inches='tight',pad_inches=0)

        plt.show(block=False)

def plotScatter(x_data, t_data):
    fig = plt.figure(figsize=(figsize_x, figsize_y))
    plt.scatter(x_data, t_data, label='All data')
    plt.xlabel('x [m]')
    plt.ylabel('t [s]')
    plt.legend(title="Training Data", bbox_to_anchor=(
        1.05, 1), loc='upper left')
    fig.tight_layout()    
    plt.show(block=False)

def plotConvergence(h, figs_dir=None, tag=''):    
    plt.figure(figsize=(10, 8))
    plt.semilogy(h.history['loss'])
    plt.title(f'Convergence {tag}')
    plt.xlabel('epochs')
    plt.ylabel('MSE loss')
    plt.grid()

    if figs_dir != None:
        path_hist_plot = os.path.join(figs_dir, f'loss_convergence_{tag}.png')
        plt.savefig(path_hist_plot)

    plt.show(block=False)