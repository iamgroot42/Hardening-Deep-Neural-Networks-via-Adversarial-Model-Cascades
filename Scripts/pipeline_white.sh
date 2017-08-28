#!/bin/bash

bbmodel=$1
epsilon=$2

mkdir -p ../Data/VanillaCNN/$epsilon
# Generate adversarial examples for whitebox model
python ../Code/generate_adversarials_fgsm.py --model_path $1 --adversary_path_x ADX.npy --adversary_path_y ADY.npy --fgsm_eps $epsilon
# Sample an adversarial image for visualization
echo "Generating images"
python ../Code/visualize_adex.py --dataset 1

