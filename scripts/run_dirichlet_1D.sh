#!/bin/sh
#BSUB -J dirichlet
#BSUB -q gpuv100
#BSUB -n 1
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -W 15:00
#BSUB -R "rusage[mem=3GB]"
#BSUB -J "../data/pinn/logs/dirichlet"

#BSUB -o ../data/pinn/logs/Output_%J.txt
#BSUB -e ../data/pinn/logs/Error_%J.txt
# -- Number of cores requested -- 
#BSUB -n 1 

# -- Notify me by email when execution begins --
#BSUB -B
# -- Notify me by email when execution ends   --
#BSUB -N

module load python3/3.8.9
module load cuda/11.1
module load cudnn/v8.0.4.30-prod-cuda-11.1
module load tensorrt/7.2.3.4-cuda-11.1

export PYTHONPATH="${PYTHONPATH}:/zhome/00/4/50173/.local/bin"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:$CUDA_ROOT/extras/CUPTI/lib64/"

python3 main_train.py --path_settings="scripts/settings/dirichlet_1D.json"