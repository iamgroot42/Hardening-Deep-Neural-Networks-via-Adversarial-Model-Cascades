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
python ../Code/generate_adversarials.py --model_path PM --adversary_path_x ADX --adversary_path_y ADY --fgsm_eps $epsilon >> ../Data/CNNSVM/$epsilon/log
# Test misclassification accuracy of proxy adversarial examples on blackbox model
python ../Code/cross_test.py --model_path BM --adversary_path_x ADX.npy --adversary_path_y ADY.npy --is_autoencoder 3 >> ../Data/CNNSVM/$epsilon/log
# Sample an adversarial image for visualization
python ../Code/visualize_adex.py --dataset 1

mv BM CNNSVM/$epsilon/
mv PM CNNSVM/$epsilon/
mv ADX.npy CNNSVM/$epsilon/
mv ADXO.npy CNNSVM/$epsilon/
mv ADY.npy CNNSVM/$epsilon/
mv PX.npy CNNSVM/$epsilon/
mv PY.npy CNNSVM/$epsilon/
mv C.pkl CNNSVM/$epsilon/
mv arch.json CNNSVM/$epsilon/
mv adv_example.png CNNSVM/$epsilon/
mv example.png CNNSVM/$epsilon/