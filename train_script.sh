#!/bin/bash
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --gres=gpu:v100:4
#SBATCH --time=23:59:00
#SBATCH --mail-user=jasonj03@connect.hku.hk
#SBATCH --mail-type=ALL
#SBATCH --account=def-xli135

cd /home/jzj/projects/rpp-xli135/jzj/ddpm-torch
module purge
module load python/3.10.13 scipy-stack opencv/4.8.1
source /home/jzj/projects/rpp-xli135/jzj/ArtificialGANFingerprints/.venv/bin/activate

# export CUDA_VISIBLE_DEVICES=0,1&&torchrun --standalone --nproc_per_node 2 --rdzv_backend c10d train.py --dataset fingerprintedlsunbedroomdataset --root /home/jzj/projects/rpp-xli135/jzj/lsun/dataset --distributed

# export CUDA_VISIBLE_DEVICES=0,1,2,3&&torchrun --standalone --nproc_per_node 4 --rdzv_backend c10d train.py --dataset customceleba --root /home/jzj/projects/rpp-xli135/jzj/datasets --distributed

export CUDA_VISIBLE_DEVICES=0,1,2,3&&torchrun --standalone --nproc_per_node 4 --rdzv_backend c10d train.py --dataset customcifar10 --root /home/jzj/projects/def-xli135/jzj/datasets --distributed

# export CUDA_VISIBLE_DEVICES=0,1,2,3&&torchrun --standalone --nproc_per_node 4 --rdzv_backend c10d train.py --dataset customcifar10 --root /home/jzj/projects/rpp-xli135/jzj/datasets --distributed

# export CUDA_VISIBLE_DEVICES=0,1,2,3&&torchrun --standalone --nproc_per_node 4 --rdzv_backend c10d train.py --dataset fingerprintedlsunbedroomdataset --root /home/jzj/projects/rpp-xli135/jzj/lsun/dataset --distributed
