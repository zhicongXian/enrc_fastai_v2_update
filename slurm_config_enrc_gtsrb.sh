#!/usr/bin/env bash
#
#SBATCH --job-name=enrc_gtsrb
#SBATCH --output=./outputs/enrc_gtsrb.txt
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
python3 ./cluster_gtsrb.py --dataset-path=/home/wiss/xian/Python_code/interpretable_multiple_clusterings/interpretable_multiple_clusterings/data/datasets/enrc_data/gtsrb  >> ./outputs/enrc_gtsrb_out.txt

