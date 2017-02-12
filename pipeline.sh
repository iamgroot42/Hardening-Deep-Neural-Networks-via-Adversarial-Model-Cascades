#!/bin/bash

epsilon=$1
perclass=$2

mkdir -p Outputs/$epsilon
# Train blackbox model
python train_model.py --is_blackbox True --save_here BM >> Outputs/$epsilon/log
# Generate training data for proxy network
python cross_test.py --model_path BM --proxy_data True --per_class_adv $perclass >> Outputs/$epsilon/log
# Train proxy model
python train_model.py --is_blackbox False --save_here PM >> Outputs/$epsilon/log
# Generate adversarial examples for proxy model
python generate_adversarials.py --model_path PM --adversary_path_x ADX.npy --adversary_path_y ADY.npy --fgsm_eps $epsilon >> Outputs/$epsilon/log
# Test misclassification accuracy of proxy adversarial examples on blackbox model
python cross_test.py --model_path BM --adversary_path_x ADX.npy --adversary_path_y ADY.npy >> Outputs/$epsilon/log
# Sample an adversarial image for visualization
python visualize_adex.py --dataset 1

mv BM Outputs/$epsilon/
mv PM Outputs/$epsilon/
mv ADX.npy Outputs/$epsilon/
mv ADXO.npy Outputs/$epsilon/
mv ADY.npy Outputs/$epsilon/
mv PX.npy Outputs/$epsilon/
mv PY.npy Outputs/$epsilon/
mv adv_example.png Outputs/$epsilon/
mv example.png Outputs/$epsilon/
