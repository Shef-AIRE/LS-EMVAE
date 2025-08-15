#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:3
#SBATCH --mem=82G
#SBATCH --cpus-per-task=10
#SBATCH --mail-user=m.suvon@sheffield.ac.uk
#SBATCH --mail-type=END,FAIL
#SBATCH --time=6:00:00
#SBATCH --output=output_%j.txt

# Load required modules
module load Anaconda3/2022.05
module load CUDA/11.8.0
module load cuDNN/8.7.0.84-CUDA-11.8.0
module load GCC/11.2.0  # Load newer C++ compiler for torch.compile

echo "Done loading module"

# Activate conda environment
source activate mvae

echo "Done loading env"


# Install dependencies
pip install torch==2.3.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install neurokit2 psutil --quiet
pip install timesfm[torch]

echo "Dependencies installed"

# Change to the working directory
export PYTHONPATH=/users/ac1xms/MICCAI2025/timesfm:$PYTHONPATH

torchrun --nproc_per_node=3 /users/ac1xms/MICCAI2025/timesfm/notebooks/Times_FM.py

echo "Done."
