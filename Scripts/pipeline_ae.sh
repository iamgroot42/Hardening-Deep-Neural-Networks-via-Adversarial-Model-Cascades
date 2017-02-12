#!/bin/bash

epsilon=$1
perclass=$2

mkdir -p ../Data/AdvOutputs/$epsilon
# Train blackbox model
python ../Code/train_model.py --is_blackbox True --save_here BM --is_autoencoder 1 >> ../Data/AdvOutputs/$epsilon/log
# Generate training data for proxy network
python ../Code/cross_test.py --model_path BM --proxy_data True --per_class_adv $perclass >> ../Data/AdvOutputs/$epsilon/log
# Train proxy model
python ../Code/train_model.py --is_blackbox False --save_here PM --is_autoencoder 1 >> ../Data/AdvOutputs/$epsilon/log
# Generate adversarial examples for proxy model
python ../Code/generate_adversarials.py --model_path PM --adversary_path_x ADX --adversary_path_y ADY --fgsm_eps $epsilon >> ../Data/AdvOutputs/$epsilon/log
# Test misclassification accuracy of proxy adversarial examples on blackbox model
python ../Code/cross_test.py --model_path BM --adversary_path_x ADX.npy --adversary_path_y ADY.npy >> ../Data/AdvOutputs/$epsilon/log
# Sample an adversarial image for visualization
#python visualize_adex.py --dataset 1

mv ../Code/BM ../Data/AdvOutputs/$epsilon/
mv ../Code/PM ../Data/AdvOutputs/$epsilon/
mv ../Code/ADX.npy ../Data/AdvOutputs/$epsilon/
mv ../Code/ADXO.npy ../Data/AdvOutputs/$epsilon/
mv ../Code/ADY.npy ../Data/AdvOutputs/$epsilon/
mv ../Code/PX.npy ../Data/AdvOutputs/$epsilon/
mv ../Code/PY.npy ../Data/AdvOutputs/$epsilon/
mv ../Code/adv_example.png ../Data/AdvOutputs/$epsilon/
mv ../Code/example.png ../Data/AdvOutputs/$epsilon/
