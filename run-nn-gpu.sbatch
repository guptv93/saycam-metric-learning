#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --gres=gpu:p40:1
#SBATCH --time=48:00:00
#SBATCH --mem=50GB
#SBATCH --job-name=nn-gpu
#SBATCH --mail-type=END
#SBATCH --mail-user=vvg239@nyu.edu
#SBATCH --output=./slurm_train_%j.out

cd /scratch/vvg239/headcam
python downstream.py
