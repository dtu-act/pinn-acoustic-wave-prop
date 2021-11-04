# ==============================================================================
# Copyright 2021 Technical University of Denmark
# Author: Nikolas Borrel-Jensen 
#
# All Rights Reserved.
#
# Licensed under the MIT License.
# ==============================================================================
import json

from models.datastructures import BoundaryCondition, BoundaryType, LossPenalties, FrequencyDependentImpedance

def parseSettings(path_to_json):
    file_handle = open(path_to_json, "r")
    data = file_handle.read()
    json_obj = json.loads(data)
    file_handle.close()

    return json_obj

def setupLossPenalties(json_obj):
    penalties_dict = json_obj['loss_penalties']

    if 'run_data' in json_obj and json_obj['run_data'] or 'run_pinn_data' in json_obj and json_obj['run_pinn_data']:
        penality_data = penalties_dict['data']
    else:
        penality_data = None

    if json_obj['boundary_type'] == 'IMPEDANCE_FREQ_DEP':
        return LossPenalties(pde=penalties_dict['pde'],ic=penalties_dict['ic'],
            bc=penalties_dict['bc'],ade=penalties_dict['ade'],data=penality_data)
    else:
        return LossPenalties(pde=penalties_dict['pde'],ic=penalties_dict['ic'],
            bc=penalties_dict['bc'],ade=None,data=penality_data)

def setupBoundaryCondition(json_obj):    
    if json_obj['boundary_type'] == 'DIRICHLET':
        return BoundaryCondition(BoundaryType.DIRICHLET, p=0)
    elif json_obj['boundary_type'] == 'NEUMANN':
        return BoundaryCondition(BoundaryType.NEUMANN, v=0)
    elif json_obj['boundary_type'] == 'IMPEDANCE_FREQ_INDEP':
        xi = json_obj['impedance_data']['xi']
        return BoundaryCondition(type=BoundaryType.IMPEDANCE_FREQ_INDEP,xi=xi)
    elif json_obj['boundary_type'] == 'IMPEDANCE_FREQ_DEP':
        Yinf = json_obj['impedance_data']['Yinf']
        A = json_obj['impedance_data']['A']
        B = json_obj['impedance_data']['B']
        C = json_obj['impedance_data']['C']
        lambdas = json_obj['impedance_data']['lambdas']
        alpha = json_obj['impedance_data']['alpha']
        beta = json_obj['impedance_data']['beta']

        impedance_data = FrequencyDependentImpedance(Yinf,A,B,C,lambdas,alpha,beta)
        return BoundaryCondition(type=BoundaryType.IMPEDANCE_FREQ_DEP,impedance_data=impedance_data)    
    else:
        raise NotImplementedError()