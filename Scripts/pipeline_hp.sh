#!/bin/bash

epsilon=$1
perclass=$2

mkdir -p ../Data/HPE/$1
# Train blackbox model
python ../Code/train_model.py --is_blackbox True --save_here BM --is_autoencoder 2 --num_clusters 64 >> ../Data/HPE/$epsilon/log
# Generate training data for proxy network
python ../Code/cross_test.py --model_path BM  --is_autoencoder 2 --proxy_data True --per_class_adv $perclass --is_autoencoder 3 >> ../Data/HPE/$epsilon/log
# Train proxy model
python ../Code/train_model.py --is_blackbox False --save_here PM --is_autoencoder 2 >> ../Data/HPE/$epsilon/log
# # Generate adversarial examples for proxy model
python ../Code/generate_adversarials.py --model_path PM --adversary_path_x ADX --adversary_path_y ADY --fgsm_eps $epsilon >> ../Data/HPE/$epsilon/log
# # Test misclassification accuracy of proxy adversarial examples on blackbox model
python ../Code/cross_test.py --is_autoencoder 2 --model_path BM --adversary_path_x ADX.npy --adversary_path_y ADY.npy >> ../Data/HPE/$epsilon/log
python ../Code/visualize_adex.py --dataset 1

mv BM ../Data/HPE/$epsilon/
mv PM ../Data/HPE/$epsilon/
mv ADX.npy ../Data/HPE/$epsilon/
mv ADXO.npy ../Data/HPE/$epsilon/
mv ADY.npy ../Data/HPE/$epsilon/
mv PX.npy ../Data/HPE/$epsilon/
mv PY.npy ../Data/HPE/$epsilon/
mv C.pkl ../Data/HPE/$epsilon/
mv adv_example.png ../Data/HPE/$epsilon/
mv example.png ../Data/HPE/$epsilon/
