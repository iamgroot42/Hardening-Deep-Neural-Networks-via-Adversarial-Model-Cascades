#!/bin/bash

epsilon=$1
mkdir -p Outputs/$epsilon
# Train blackbox model
python train_model.py --is_blackbox True --save_here BM >> Outputs/$epsilon/log
# Train proxy model
python train_model.py --is_blackbox False --save_here PM >> Outputs/$epsilon/log
# Generate adversarial examples for proxy model
python generate_adversarials.py --model_path PM --adversary_path_x ADX --adversary_path_y ADY --fgsm_eps $epsilon >> Outputs/$epsilon/log
# Test misclassification accuracy of proxy adversarial examples on blackbox model
python cross_test.py --model_path BM --adversary_path_x ADX.npy --adversary_path_y ADY.npy >> Outputs/$epsilon/log
# Sample an adversarial image for visualization
python visualize_adex.py

mv BM Outputs/$epsilon/
mv PM Outputs/$epsilon/
mv ADX.npy Outputs/$epsilon/
mv ADY.npy Outputs/$epsilon/
mv adv_example.png Outputs/$epsilon/
