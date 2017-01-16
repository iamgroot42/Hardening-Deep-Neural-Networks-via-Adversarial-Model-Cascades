#!/bin/bash

mkdir -p Outputs/$1
epsilon=$1
# Train blackbox model
python train_model.py --is_blackbox True --save_here BM >> Outputs/"$1"/log
# Train proxy model
python train_model.py --is_blackbox False --save_here PM >> Outputs/"$1"/log
# Generate adversarial examples for proxy model
python generate_adversarials.py --model_path PM --adversary_path_x ADX --adversary_path_y ADY --fgsm_eps "$1" >> Outputs/"$1"/log
# Test misclassification accuracy of proxy adversarial examples on blackbox model
python cross_test.py --model_path BM --adversary_path_x ADX.npy --adversary_path_y ADY.npy >> Outputs/"$1"/log
# Sample an adversarial image for visualization
python visualize_adex.py

mv BM Outputs/$1/
mv PM Outputs/$1/
mv ADX.npy Outputs/$1/
mv ADY.npy Outputs/$1/
mv adv_example.png Outputs/$1/
