# ==============================================================================
# Copyright 2021 Technical University of Denmark
# Author: Nikolas Borrel-Jensen 
#
# All Rights Reserved.
#
# Licensed under the MIT License.
# ==============================================================================
import tensorflow as tf
import numpy as np
# https://keras.io/api/layers/initializers/
class SineInitializer(tf.keras.initializers.Initializer):
    def __call__(self, shape, dtype=None):
        num_input = shape[0]

        return tf.random.uniform(shape,
            minval=-np.sqrt(6 / num_input) / 30,
            maxval= np.sqrt(6 / num_input) / 30, 
            dtype=dtype)

def first_layer_sine_init(functional, dtype=None):
    ws = functional.layers[0].get_weights()
    w = ws[0]

    shape = w.shape
    num_input = shape[0]

    w_init = np.random.uniform(
        low= -1 / num_input,
        high= 1 / num_input, 
        size=shape)

    ws[0] = w_init
    functional.layers[0].set_weights(ws)

