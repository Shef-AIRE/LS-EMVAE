#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=400G
#SBATCH --cpus-per-task=40
#SBATCH --mail-user=m.suvon@sheffield.ac.uk
#SBATCH --mail-type=END,FAIL
#SBATCH --time=24:00:00
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
# Install dependencies
pip install torch==2.0.1+cu118 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install neurokit2 psutil --quiet

echo "Dependencies installed"

# Change to the working directory
cd /users/ac1xms/MICCAI2025/

# Run the Python script
python -u 12LS_EMVAE_with_reg_w_o_MoE.py

echo "Done."
