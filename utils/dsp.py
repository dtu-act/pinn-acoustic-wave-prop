# ==============================================================================
# Copyright 2021 Technical University of Denmark
# Author: Nikolas Borrel-Jensen 
#
# All Rights Reserved.
#
# Licensed under the MIT License.
# ==============================================================================
from scipy.fftpack import fft
import numpy as np
 
def get_fft_values(y, fs, NFFT=None):
    
    if NFFT == None:
        N = len(y)        
    else:
        N = np.maximum(NFFT,len(y))
    
    Y = fft(y, n=NFFT)
    
    freqs_compare = fs*np.linspace(0, N//2-1, N//2)/N
    freqs = fs*np.array([i for i in range(N//2)])/N
    assert((freqs_compare == freqs).all())

    Y = 2.0/N * np.abs(Y[0:N//2])
    return freqs, Y