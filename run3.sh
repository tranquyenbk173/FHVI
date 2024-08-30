#!/bin/bash -e

#SBATCH --job-name=abcxyz # create a short name for your job
#SBATCH --output=/home/quyentt15/quyentt15/Bayesian_finetuning/mbpp%A.out # create a output file
#SBATCH --error=/home/quyentt15/quyentt15/Bayesian_finetuning/mbpp%A.err # create a error file
#SBATCH --partition=research # choose partition
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-gpu=40GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=10-00:00          # total run time limit (DD-HH:MM)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail          # send email when job fails
#SBATCH --mail-user=v.quyentt15@vinai.io

/lustre/scratch/client/vinai/users/quyentt15/envs/anaconda3/bin/conda activate coda
cd /home/quyentt15/quyentt15/Bayesian_finetuning/

python main.py fit --config configs/lora/cifar100-r16-lr-0.05_svgd_copy_2.yaml

# python main.py fit --config configs/lora/cifar100-r16-lr-0.05_svgd_copy.yaml