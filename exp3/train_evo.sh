#!/bin/bash
#SBATCH -o exp3%j.out
#SBATCH -e exp3%j.err

# Activate virtual environment
source ~/EvoRob/VENV/bin/activate

cd ~/EvoRob/exp3


python evolution.py --mode train-pygad --population-size 100 --generations 400  --episodes 4 --max-steps 500 --save robot_pygad_torch.npzpygad.GA
