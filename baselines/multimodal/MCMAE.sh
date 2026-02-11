#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=400G
#SBATCH --cpus-per-task=40
#SBATCH --mail-user=m.suvon@sheffield.ac.uk
#SBATCH --mail-type=END,FAIL
#SBATCH --time=96:00:00
#SBATCH --output=output_%j.txt

module load Anaconda3/2022.05
module load CUDA/11.8.0
module load cuDNN/8.7.0.84-CUDA-11.8.0
module load GCC/11.2.0

echo "Done loading module"

source activate mvae

echo "Done loading env"

pip install torch==2.0.1+cu118 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install neurokit2 psutil --quiet

echo "Dependencies installed"

cd /users/ac1xms/MICCAI2025/

python -u MCMAE.py

echo "Done."
