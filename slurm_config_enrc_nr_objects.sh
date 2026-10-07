#!/usr/bin/env bash
#
#SBATCH --job-name=enrc_nr_objects
#SBATCH --output=./outputs/deep_imc_nr_objects.txt
#SBATCH --ntasks=1
#SBATCH --time=10-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=128G

# debug info
hostname
which python3
nvidia-smi

env

# venv
source /home/wiss/xian/venvs/subspace_clustering_3_12/bin/activate
export BLAS=/usr/lib/x86_64-linux-gnu/blas/libblas.so.3
export LAPACK=/usr/lib/x86_64-linux-gnu/lapack/liblapack.a
# pip install -U pip setuptools wheel
# train
# --TODO do not forget to install
python3 ./cluster_nrobjects_new.py --dataset-path=data/datasets/enrc_data/nr_objects  >> ./outputs/enrc_nr_objects_out.txt

