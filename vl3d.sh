#!/bin/bash

# AUTHOR : Alberto M. Esmoris Pena
# Launcher-wrapper for virtualearn3D for bash

# The python binary
PYBIN='python3'

# The call
PYTHONPATH=${PWD}:${PYTHONPATH} ${PYBIN} src/vl3d.py ${@}
