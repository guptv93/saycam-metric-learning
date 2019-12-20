#!/bin/bash

#SBATCH --job-name=nn-gpu
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=40GB
#SBATCH --time=24:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=vvg239@nyu.edu

cd /misc/vlgscratch4/LakeGroup/vaibhav/saycam-metric-learning
python -u  downstream.py
