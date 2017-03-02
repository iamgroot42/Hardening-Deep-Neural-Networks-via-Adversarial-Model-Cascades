#!/bin/bash

epsilon=$1
perclass=$2

mkdir -p ../Data/Outputs/$epsilon
# Train blackbox model
python ../Code/train_model.py --is_blackbox True --save_here BM --specialCNN separable >> ../Data/Outputs/$epsilon/log
# Generate training data for proxy network
python ../Code/cross_test.py --model_path BM --proxy_data True --per_class_adv $perclass >> ../Data/Outputs/$epsilon/log
# Train proxy model
python ../Code/train_model.py --is_blackbox False --save_here PM >> ../Data/Outputs/$epsilon/log
# Generate adversarial examples for proxy model
python ../Code/generate_adversarials.py --model_path PM --adversary_path_x ADX.npy --adversary_path_y ADY.npy --fgsm_eps $epsilon >> ../Data/Outputs/$epsilon/log
# Test misclassification accuracy of proxy adversarial examples on blackbox model
python ../Code/cross_test.py --model_path BM --adversary_path_x ADX.npy --adversary_path_y ADY.npy >> ../Data/Outputs/$epsilon/log
# Sample an adversarial image for visualization
python ../Code/visualize_adex.py --dataset 1

mv BM ../Data/Outputs/$epsilon/
mv PM ../Data/Outputs/$epsilon/
mv ADX.npy ../Data/Outputs/$epsilon/
mv ADXO.npy ../Data/Outputs/$epsilon/
mv ADY.npy ../Data/Outputs/$epsilon/
mv PX.npy ../Data/Outputs/$epsilon/
mv PY.npy ../Data/Outputs/$epsilon/
mv adv_example.png ../Data/Outputs/$epsilon/
mv example.png ../Data/Outputs/$epsilon/
