# ==============================================================================
# Copyright 2021 Technical University of Denmark
# Author: Nikolas Borrel-Jensen 
#
# All Rights Reserved.
#
# Licensed under the MIT License.
# ==============================================================================
import numpy as np
import sciann as sn
from typing import Callable, Tuple

def sciann_gaussianSource(c: float, fn: float, c_denorm=1) -> Tuple[Callable, float]:
    sigma_t = 1/(sn.math.pi*fn/2)
    sigma_x = c/(sn.math.pi*fn/2)
    mu = 4*sigma_t # smooth initialization

    source_t = lambda t: sn.math.exp(-(t/c_denorm-mu)**2/(sigma_t**2))
    source_x = lambda x,x0: sn.math.exp(-(x-x0)**2/(sigma_x**2))

    source = lambda x,t,x0: source_t(t)*source_x(x,x0)

    return source, mu

def gaussianSource(c: float, fn: float) -> Tuple[Callable, float]:
    sigma_t = 1/(np.pi*fn/2)
    sigma_x = c/(np.pi*fn/2)
    mu = 4*sigma_t # smooth initialization

    source_t = lambda t: np.exp(-(t-mu)**2/(sigma_t**2))
    source_x = lambda x,x0: np.exp(-(x-x0)**2/(sigma_x**2))

    source = lambda x,t,x0: source_t(t)*source_x(x,x0)

    return source, source_t, mu

def sciann_gaussianSourceDiff(c: float, fn: float) -> Tuple[Callable, float]:
    sigma_t = 1/(sn.math.pi*fn/2)
    sigma_x = c/(sn.math.pi*fn/2)
    mu = 4*sigma_t # smooth initialization
    
    source_t = lambda t: -(t - mu)/sigma_t*sn.math.exp( - (t - mu)**2/(sigma_t**2) )
    source_x = lambda x,x0: -(x - x0)/sigma_x*sn.math.exp( - (x - x0)**2/(sigma_x**2) )

    source = lambda x,t: source_t(t)*source_x(x)

    return source, mu

def sciann_gaussianIC(sigma0: float) -> Callable:
    source_ic = lambda x,x0: sn.math.exp(-0.5*((x-x0)/sigma0)**2)
    return source_ic

def sciann_gaussianIC_fixed(sigma0: float, x0: float) -> Callable:
    source_ic = lambda x: sn.math.exp(-0.5*((x-x0)/sigma0)**2)
    return source_ic