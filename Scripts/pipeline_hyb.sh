#!/bin/bash

epsilon=$1
perclass=$2

mkdir -p ../Data/CNNSVM/$epsilon
# Train blackbox model
python ../Code/train_model.py --is_blackbox True --save_here BM --is_autoencoder 3 >> ../Data/CNNSVM/$epsilon/log
# Generate training data for proxy network
python ../Code/cross_test.py --model_path BM --proxy_data True --per_class_adv $perclass --is_autoencoder 3 >> ../Data/CNNSVM/$epsilon/log
# Train proxy model
python ../Code/train_model.py --is_blackbox False --save_here PM --is_autoencoder 3 >> ../Data/CNNSVM/$epsilon/log
# Generate adversarial examples for proxy model
python ../Code/generate_adversarials_fgsm.py --model_path PM --adversary_path_x ADX.npy --adversary_path_y ADY.npy --fgsm_eps $epsilon >> ../Data/CNNSVM/$epsilon/log
# Test misclassification accuracy of proxy adversarial examples on blackbox model
python ../Code/cross_test.py --model_path BM --adversary_path_x ADX.npy --adversary_path_y ADY.npy --is_autoencoder 3 >> ../Data/CNNSVM/$epsilon/log
# Sample an adversarial image for visualization
python ../Code/visualize_adex.py --dataset 1

mv ADX.npy ../Data/CNNSVM/$epsilon/
mv ADXO.npy ../Data/CNNSVM/$epsilon/
mv ADY.npy ../Data/CNNSVM/$epsilon/
mv PX.npy ../Data/CNNSVM/$epsilon/
mv PY.npy ../Data/CNNSVM/$epsilon/
mv adv_example.png ../Data/CNNSVM/$epsilon/
mv example.png ../Data/CNNSVM/$epsilon/
