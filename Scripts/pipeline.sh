#!/bin/bash

epsilon=$1
perclass=$2

mkdir -p ../Data/VanillaCNN/$epsilon
# Train blackbox model
python ../Code/train_model.py --is_blackbox True --save_here BM --specialCNN sota --batch_size 16 --learning_rate 0.1 --nb_epochs 300 >> ../Data/VanillaCNN/$epsilon/log
# Generate training data for proxy network
python ../Code/cross_test.py --model_path BM --proxy_data True --per_class_adv $perclass >> ../Data/VanillaCNN/$epsilon/log
# Train proxy model
python ../Code/train_model.py --is_blackbox False --save_here PM --batch_size 16 --learning_rate 0.01 --nb_epochs 500 >> ../Data/VanillaCNN/$epsilon/log
# Generate adversarial examples for proxy model
python ../Code/generate_adversarials.py --model_path PM --adversary_path_x ADX.npy --adversary_path_y ADY.npy --fgsm_eps $epsilon >> ../Data/VanillaCNN/$epsilon/log
# Test misclassification accuracy of proxy adversarial examples on blackbox model
python ../Code/cross_test.py --model_path BM --adversary_path_x ADX.npy --adversary_path_y ADY.npy >> ../Data/VanillaCNN/$epsilon/log
# Sample an adversarial image for visualization
python ../Code/visualize_adex.py --dataset 1

mv ADX.npy ../Data/VanillaCNN/$epsilon/
mv ADXO.npy ../Data/VanillaCNN/$epsilon/
mv ADY.npy ../Data/VanillaCNN/$epsilon/
mv PX.npy ../Data/VanillaCNN/$epsilon/
mv PY.npy ../Data/VanillaCNN/$epsilon/
mv adv_example.png ../Data/VanillaCNN/$epsilon/
mv example.png ../Data/VanillaCNN/$epsilon/
