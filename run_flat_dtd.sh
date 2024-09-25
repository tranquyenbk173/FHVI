#!/bin/bash -e

#SBATCH --job-name=dtd_deepEns # create a short name for your job
#SBATCH --output=/lustre/scratch/client/vinai/users/quanpn2/Bayesian_finetuning/sbatch_results/mbpp%A.out # create a output file
#SBATCH --error=/lustre/scratch/client/vinai/users/quanpn2/Bayesian_finetuning/sbatch_results/mbpp%A.err # create a error file
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
#SBATCH --mail-user=v.quanpn2@vinai.io



module purge
module load python/miniconda3/miniconda3
eval "$(conda shell.bash hook)"



conda activate /lustre/scratch/client/vinai/users/quanpn2/angry

# Run Python script
python main.py fit --config configs/lora/flat/vtab_dtd-r16-lr0.05_flat.yaml
