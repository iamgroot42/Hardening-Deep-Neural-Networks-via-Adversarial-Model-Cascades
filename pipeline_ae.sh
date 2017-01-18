#!/bin/bash

mkdir -p AdvOutputs/$1
epsilon=$1
# Train blackbox model
python train_model.py --is_blackbox True --save_here BM --is_autoencoder 1 >> AdvOutputs/"$1"/log
# Train proxy model
python train_model.py --is_blackbox False --save_here PM --is_autoencoder 1 >> AdvOutputs/"$1"/log
# Generate adversarial examples for proxy model
python generate_adversarials.py --model_path PM --adversary_path_x ADX --adversary_path_y ADY --fgsm_eps "$1" >> AdvOutputs/"$1"/log
# Test misclassification accuracy of proxy adversarial examples on blackbox model
python cross_test.py --model_path BM --adversary_path_x ADX.npy --adversary_path_y ADY.npy >> AdvOutputs/"$1"/log
# Sample an adversarial image for visualization
python visualize_adex.py --dataset 1

mv BM AdvOutputs/$1/
mv PM AdvOutputs/$1/
mv ADX.npy AdvOutputs/$1/
mv ADY.npy AdvOutputs/$1/
mv adv_example.png AdvOutputs/$1/
