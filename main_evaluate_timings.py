# ==============================================================================
# Copyright 2021 Technical University of Denmark
# Author: Nikolas Borrel-Jensen 
#
# All Rights Reserved.
#
# Licensed under the MIT License.
# ==============================================================================
import sys
sys.path.append('..') # needed unless installing forked lib from github

import argparse
import run.run_evaluate_timings as pinn_model

parser = argparse.ArgumentParser()
parser.add_argument("--path_settings", type=str, required=True)
parser.add_argument("--trained_model_tag", type=str, required=True)
args = parser.parse_args()

if __name__ == "__main__":
  settings_path = args.path_settings
  trained_model_tag = args.trained_model_tag
  pinn_model.evaluate(settings_path, trained_model_tag)