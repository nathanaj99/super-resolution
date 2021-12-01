#!/bin/bash
#SBATCH --begin=now
#SBATCH --job-name=tf_gpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nathanjo@law.stanford.edu
#SBATCH --partition=owners
#SBATCH --time=1:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:2

cd ../

singularity exec --nv $GROUP_HOME/singularity/bva_tf.sif python edsr.py
