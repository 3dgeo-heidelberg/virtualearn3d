#!/bin/bash

# ------------------------------------------------------------------#
# AUTHOR: Alberto M. Esmoris Pena                                   #
# BRIEF: Configure environment to run VL3D at CESGA FT-III          #
#                                                                   #
# NOTE that the conda environment must created from a GPU node to   #
# enable GPU acceleration for deep learning.                        #
# ------------------------------------------------------------------#



# ---  GLOBAL VARIABLES  --- #
# -------------------------- #
export VL3D_DIR='/home/usc/ci/aep/git/virtualearn3d/'
export VL3D_SCRIPT='/home/usc/ci/aep/git/virtualearn3d/vl3d.py'
export VL3D_ENV="${STORE2}/vl3d_conda_env"



# ---  CONFIGURE ENVIRONMENT  --- #
# ------------------------------- #
# Load modules
module load cesga/system miniconda3/22.11.1-1
# Activate conda environment
conda activate "${VL3D_ENV}"
# Link ptxas
export XLA_FLAGS=--xla_gpu_cuda_data_dir=${CONDA_PREFIX}


# ---  UTIL FUNCTIONS  --- #
# ------------------------ #
# Change working directory to VL3D directory
function cd_vl3d {
    cd "${VL3D_DIR}"
}
