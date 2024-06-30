#!/bin/bash
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --gres=gpu:v100l:1
#SBATCH --time=5:59:00
#SBATCH --mail-user=jasonj03@connect.hku.hk
#SBATCH --mail-type=ALL
#SBATCH --account=def-xli135

cd /home/jzj/projects/rpp-xli135/jzj/ddpm-torch
module purge
module load python/3.10.13 scipy-stack opencv/4.8.1
source /home/jzj/projects/rpp-xli135/jzj/ddpm-torch/.venv/bin/activate

python generate.py --dataset multifingerprintdataset --chkpt-path ./chkpts/multifingerprintdataset/multifingerprintdataset_500.pt --use-ddim --skip-schedule quadratic --subseq-size 100 --suffix _epch500_ddim --num-gpus 1 --total-size 4096 --batch-size 128
