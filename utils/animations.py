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
import matplotlib.animation as anim
from matplotlib.gridspec import GridSpec

# https://towardsdatascience.com/intro-to-dynamic-visualization-with-python-animations-and-interactive-plots-f72a7fb69245
# https://stackoverflow.com/questions/10388462/matplotlib-different-size-subplots

def animateAccumulators(t, x, p_pred, p_ref, acc_pred_l, acc_pred_r, acc_ref_l, acc_ref_r, title, labels_acc, figs_dir=None, tag=''):
    fig = plt.figure(figsize=(28,24))
    fig.suptitle(title, fontsize=30)

    N_acc = len(acc_pred_l)
    gs = GridSpec(N_acc+1, 2, figure=fig)
    
    axes = []

    for i in range(2):
        for j in range(N_acc):
            axes.append(fig.add_subplot(gs[j,i]))
    
    axes.append(plt.subplot(gs[-1,:])) # the spatial wave propagation: last row, span two cols
    
    linewidth = 2

    lines_p_pred = []
    lines_p_ref = []
    lines_pred_l = []
    lines_ref_l = []
    lines_pred_r = []
    lines_ref_r = []

    for i in range(N_acc):
        label_str = labels_acc[i]    
        ax = axes[i]

        line = ax.plot([], [], linewidth=linewidth, linestyle='--', color='red', label='ref')
        lines_ref_l.extend(line)

        line = ax.plot([], [], linewidth=linewidth, linestyle='-', color='blue', label='pred')
        lines_pred_l.extend(line)
        
        ax.set_xlim([0, max(t)])
        ax.set_ylim([min(acc_ref_l[i]), max(acc_ref_l[i])])
        ax.set_ylabel(label_str, fontsize=15)        
        ax.set_xlabel('time [sec]', fontsize=15)

        if i == 0:
            ax.title.set_text('left')
            ax.title.set_size(20)
            ax.legend(handles=[lines_ref_l[-1], lines_pred_l[-1]], loc='upper right')

    for i in range(N_acc):
        ax = axes[N_acc+i]

        line = ax.plot([], [], linewidth=linewidth, linestyle='--', color='red', label='ref')
        lines_ref_r.extend(line)

        line = ax.plot([], [], linewidth=linewidth, linestyle='-', color='blue', label='pred')
        lines_pred_r.extend(line)
        
        ax.set_xlim([0, max(t)])
        ax.set_ylim([min(acc_ref_r[i]), max(acc_ref_r[i])])
        ax.set_ylabel(label_str, fontsize=15)
        ax.set_xlabel('time [sec]', fontsize=15)

        if i == 0:
            ax.title.set_text('right')
            ax.title.set_size(20)
            ax.legend(handles=[lines_ref_l[-1], lines_pred_l[-1]], loc='upper right')

    ax = axes[-1]
    ax.title.set_text('wave propagation')
    ax.title.set_size(20)
    ax.set_xlim([-1, 1])
    ax.set_ylim([0, 1])    

    line = ax.plot([], [], linewidth=linewidth, linestyle='--', color='red', label='ref')            
    lines_p_ref.extend(line)

    line = ax.plot([], [], linewidth=linewidth, linestyle='-', color='blue', label='pred')
    lines_p_pred.extend(line)
    
    ax.set_xlabel('length [m]', fontsize=15)
    ax.set_ylabel('pressure [Pa]', fontsize=15)
    ax.legend(handles=[lines_p_ref[-1], lines_p_pred[-1]], loc='upper right')
    
    def init():
        #init lines
        for line in lines_ref_l:
            line.set_data([], [])
        for line in lines_pred_l:
            line.set_data([], [])      
        for line in lines_ref_r:
            line.set_data([], [])
        for line in lines_pred_r:
            line.set_data([], [])
        for line in lines_p_ref:
            line.set_data([], [])
        for line in lines_p_pred:
            line.set_data([], [])

        return np.concatenate((lines_ref_l,lines_pred_l,lines_ref_r,lines_pred_r))

    def animate(j):
        for i,line in enumerate(lines_ref_l):
            line.set_data(t[0:j], acc_ref_l[i][0:j])
        for i,line in enumerate(lines_pred_l):
            line.set_data(t[0:j], acc_pred_l[i][0:j])
        for i,line in enumerate(lines_ref_r):
            line.set_data(t[0:j], acc_ref_r[i][0:j])
        for i,line in enumerate(lines_pred_r):
            line.set_data(t[0:j], acc_pred_r[i][0:j])
        for _,line in enumerate(lines_p_ref):
            line.set_data(x, p_ref[j])
        for _,line in enumerate(lines_p_pred):
            line.set_data(x, p_pred[j])        

        return np.concatenate((lines_ref_l,lines_pred_l,lines_ref_r,lines_ref_r,lines_p_ref,lines_p_pred))

    ani = anim.FuncAnimation(fig, animate, init_func=init, frames=len(t), blit=True, repeat=False)

    if figs_dir != None:
        path_plot = os.path.join(figs_dir, f'animation_acc_{tag}.mp4')
        ani.save(path_plot)

    plt.show(block=False)    

def plotAnimation(x_mesh, t_mesh, p_pred, p_exact, figs_dir=None, plotnth=1, tag=''):
    t = np.unique(t_mesh[::plotnth])
    x = np.unique(x_mesh[::plotnth])

    p_pred_mat = p_pred[::plotnth].reshape(len(t),len(x))
    p_exact_mat = p_exact[::plotnth].reshape(len(t),len(x))    

    plt.figure(figsize=(10,6))
    ax1 = plt.subplot(3,2,1)
    ax2 = plt.subplot(3,2,2)
    ax3 = plt.subplot(3,2,3)
    ax4 = plt.subplot(3,2,4)
    ax5 = plt.subplot(3,1,2)
    axes = [ax1,ax2,ax3,ax4,ax5]

    for i,ax in axes:
        ax.xlim([-1, 1])
        ax.ylim([0, 1])
        ax.title('$Prediction, NN(x,t,x0)\simeq p(x,t,x0)$')
        ax.xlabel('x [m]')
        ax.ylabel('p [pa]')

    legends = ['NN prediction', 'Reference']
    line_params = [[4,'-','blue'], [4,'--','red']]
    N=len(legends)

    lines = [ax5.plot([], [], linewidth=line_params[i][0], linestyle=line_params[i][1], color=line_params[i][2], 
        label=legends[i])[0] for i in range(N)] #lines to animate

    lines_rec = [axes[i].plot([], [], linewidth=line_params[i][0], linestyle=line_params[i][1], color=line_params[i][2], 
        label=legends[i])[0] for i in range(axes)-1] #lines to animate
    
    def init():
        #init lines
        for line in lines:
            line.set_data([], [])

        for line in lines_rec:
            line.set_data([], [])

        return lines #return everything that must be updated

    def animate(i):
        lines[0].set_data(x, p_pred_mat[i])
        lines[1].set_data(x, p_exact_mat[i])
        
        return lines
    
    fig = plt.gcf()   

    ani = anim.FuncAnimation(fig, animate, init_func=init, frames=len(t), blit=True, repeat=False)    

    plt.legend(handles=[lines[0], lines[1]], loc='upper right')

    if figs_dir != None:
        path_plot = os.path.join(figs_dir, f'p_anim_{tag}.mp4')
        ani.save(path_plot)

    plt.show(block=False)

def plotAnimations(x, t, p_preds, p_refs, p_ir_preds, p_ir_refs, xs0, rs0, figs_dir=None,title=''):

    fig = plt.figure(figsize=(32,18))
    fig.suptitle(title, fontsize=30)

    N_rec = len(rs0)
    axes = []

    for i in range(N_rec):
        for j in range(3):
            axes.append(plt.subplot(3,N_rec,j*N_rec+i + 1))
    
    linewidth = 2

    lines_pred = []
    lines_ref = []
    lines_recs_pred = []
    lines_recs_ref = []
    lines_recs_errs = []

    for i,ax in enumerate(axes):
        if np.mod(i,3) == 0:            
            line_rec = ax.axvline(x=rs0[i//3], color='blue', linestyle='--', label='receiver')

            ax.title.set_text(f'$x_0 = {xs0[i//3]}$ m')
            ax.title.set_size(20)

            ax.set_xlim([-1, 1])
            ax.set_ylim([0, 1])

            line = ax.plot([], [], linewidth=linewidth, linestyle='--', color='red', label='ref')            
            lines_ref.extend(line)

            line = ax.plot([], [], linewidth=linewidth, linestyle='-', color='blue', label='pred')
            lines_pred.extend(line)
            
            if i == 0:
                ax.set_xlabel('length [m]', fontsize=15)
                ax.set_ylabel('pressure [Pa]', fontsize=15)
                ax.legend(handles=[lines_ref[-1], lines_pred[-1], line_rec], loc='upper right')
            else:
                #ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
        else:
            ax.set_xlim([0, max(t)])

            if np.mod(i-1,3) == 0:
                ax.set_ylim([0, 0.5])           
                line = ax.plot([], [], linewidth=linewidth, linestyle='--', color='red', label='IR ref')                
                lines_recs_ref.extend(line)
                line = ax.plot([], [], linewidth=linewidth, linestyle='-', color='blue', label='IR pred')
                lines_recs_pred.extend(line)
                if i == 1 or i == 2:     
                    ax.set_xlabel('time [sec]', fontsize=15)
                    ax.set_ylabel('pressure [Pa]', fontsize=15)
                    ax.legend(handles=[lines_recs_ref[-1], lines_recs_pred[-1]], loc='upper right')
                else:
                    #ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
                
            elif np.mod(i-2,3) == 0:
                ax.set_ylim([0, 0.0125])
                line = ax.plot([], [], linewidth=linewidth, linestyle='-', color='orange', label='err')
                lines_recs_errs.extend(line)
                if i == 1 or i == 2:     
                    ax.set_xlabel('time [sec]')
                    ax.set_ylabel('pressure [Pa]')
                    ax.legend(handles=line, loc='upper right')
                else:
                    #ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
    
    def init():
        #init lines
        for line in lines_pred:
            line.set_data([], [])
        for line in lines_ref:
            line.set_data([], [])
        for line in lines_recs_pred:
            line.set_data([], [])
        for line in lines_recs_ref:
            line.set_data([], [])
        for line in lines_recs_errs:
            line.set_data([], [])

        return np.concatenate((lines_pred,lines_ref,lines_recs_pred,lines_recs_ref,lines_recs_errs))

    def animate(j):
        for i,line in enumerate(lines_pred):
            line.set_data(x, p_preds[i][j])
        for i,line in enumerate(lines_ref):
            line.set_data(x, p_refs[i][j])
        for i,line in enumerate(lines_recs_pred):
            line.set_data(t[0:j], p_ir_preds[i][0:j])
        for i,line in enumerate(lines_recs_ref):
            line.set_data(t[0:j], p_ir_refs[i][0:j])
        for i,line in enumerate(lines_recs_errs):
            errs = abs(p_ir_preds[i][0:j] - p_ir_refs[i][0:j])
            line.set_data(t[0:j], errs)

        return np.concatenate((lines_pred,lines_ref,lines_recs_pred,lines_recs_ref,lines_recs_errs))

    ani = anim.FuncAnimation(fig, animate, init_func=init, frames=len(t), blit=True, repeat=False)

    if figs_dir != None:
        path_plot = os.path.join(figs_dir, f'animation_all.mp4')
        ani.save(path_plot)

    plt.show(block=False)