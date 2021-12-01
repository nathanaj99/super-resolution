#!/bin/bash
#SBATCH --begin=now
#SBATCH --job-name=tf_gpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nathanjo@law.stanford.edu
#SBATCH --partition=owners
#SBATCH --time=1:00:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:2

cd ../

singularity exec --nv $GROUP_HOME/singularity/tf_test.sif python edsr.py --image_path /oak/stanford/groups/deho/building_compliance/berkeley_naip_2020/berkeley_ne.tif --resolution_factor 3 --out_file /oak/stanford/groups/deho/building_compliance/berkeley_naip_2020_superres/berkeley_ne.tif
