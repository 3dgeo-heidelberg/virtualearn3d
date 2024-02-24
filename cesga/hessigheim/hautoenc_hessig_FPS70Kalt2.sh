#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gres=gpu:a100
#SBATCH -t 96:00:00
#SBATCH --mem 246GB
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=albertoesmp@gmail.com

# Author: Alberto M. Esmoris Pena
# Brief: Script to train models with VL3D framework


# ---  PREPARE ENVIRONMENT  --- #
# ----------------------------- #
source /home/usc/ci/aep/git/virtualearn3d/cesga/vl3d_cesga_env.sh


# ---  VARIABLES  --- #
# ------------------- #
TRAINING_SPEC='/home/usc/ci/aep/git/virtualearn3d/cesga/hessigheim/hautoenc_hessig_train_X_FPS70Kalt2.json'
PREDICTIVE_SPEC='/home/usc/ci/aep/git/virtualearn3d/cesga/hessigheim/hautoenc_hessig_predict_X_FPS70Kalt2.json'

# ---  EXECUTION  --- #
# ------------------- #
# RUN SCRIPTS
cd_vl3d
srun python ${VL3D_SCRIPT} --test
srun python ${VL3D_SCRIPT} --pipeline ${TRAINING_SPEC}
srun python ${VL3D_SCRIPT} --pipeline ${PREDICTIVE_SPEC}
