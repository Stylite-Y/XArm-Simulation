#!/bin/bash

# RAISIMUNITYPATH='/home/stylite-y/Documents/Raisim/raisim_workspace/raisimlib/raisimUnity/linux'
RAISIMUNITYPATH='/home/stylite-y/Documents/Raisim/raisim_workspace/raisimlib/raisimUnity/linux'
WORKPATH='/home/stylite-y/Documents/Master/Manipulator/Simulation/model_raisim'

cd $RAISIMUNITYPATH
./raisimUnity.x86_64 &

cd $WORKPATH
python3.6 LISM.py
# python