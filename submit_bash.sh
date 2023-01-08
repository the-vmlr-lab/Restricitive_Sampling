#!/bin/bash

#SBATCH -p gpu20
#SBATCH -t 0-12:00:00
#SBATCH -o output/test2.out
#SBATCH -e output/test2.out
#SBATCH -a 1
#SBATCH --gres gpu:1

# setup the slurm


#start training
echo "Run the python"
echo $PWD

python training_script.py  -m 1 -lp 5 -save_folder=“results” -model_name=‘base2-m1-lp5’ -epochs=100


