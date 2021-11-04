# ==============================================================================
# Copyright 2021 Technical University of Denmark
# Author: Nikolas Borrel-Jensen 
#
# All Rights Reserved.
#
# Licensed under the MIT License.
# ==============================================================================
import shutil
import tensorflow as tf
import os
import time
import numpy as np
from setup.settings import Settings
from setup.setup_nn import  setupPinnModelFromWeights
import setup.parsers as parsers

def evaluate(settings_path, model_tag):
    receiver_pos = 0.5

    # SETUP
    settings_dict = parsers.parseSettings(settings_path)
    settings = Settings(settings_dict)
    settings.dirs.createDirs(delete_existing=True)
    shutil.copyfile(settings_path, os.path.join(settings.dirs.id_dir, 'settings.json')) # copy settings

    fs = 44100/settings.physics.c_phys
    t_data = np.linspace(0,settings.domain.tmax,round(fs*settings.domain.tmax))
    x_data = np.array([receiver_pos]*len(t_data))
    x0_data = np.array(settings.domain.x0_sources*len(t_data))

    checkpoint_path = os.path.join(settings.dirs.transfer_models_dir, model_tag)
    latest = tf.train.latest_checkpoint(checkpoint_path)
    if latest == None:
        raise FileNotFoundError(f'Weights not found: %s', checkpoint_path)
    m, funcs, _ = setupPinnModelFromWeights(latest, settings)

    total_time = 0
    N = 100
    i = 0
    while i < N:
        start_time = time.perf_counter()
        _ = funcs.p.eval(m, [x_data, t_data, x0_data])
        end_time = time.perf_counter()
        total_time += end_time - start_time
        i += 1

    evaluation_time = total_time/N

    out_path = os.path.join(settings.dirs.id_dir, 'timings.txt')
    with open(out_path, 'w') as f:
        f.write('----------------------\n')
        f.write('Runtime measurements of the surrogate model:\n')
        f.write('----------------------\n')
        f.write(f'#layers (pde) = {settings.network.p_nn.num_layers}\n')
        f.write(f'#neurons (pde) = {settings.network.p_nn.num_neurons}\n')
        f.write(f'simulation time = {settings.domain.tmax}\n')
        f.write(f'fmax = {settings.physics.fmax}\n')
        f.write(f'#samples = {len(t_data)}\n\n')
        f.write('**evaluation time**: %e\n' % evaluation_time)
        f.write('----------------------')