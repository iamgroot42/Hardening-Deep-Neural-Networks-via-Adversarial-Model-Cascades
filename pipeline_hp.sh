#!/bin/bash

epsilon=$1

mkdir -p HPE/$1
# Train blackbox model
python train_model.py --is_blackbox True --save_here BM --is_autoencoder 2 --num_clusters 64 #>> HPE/$1/log
# Train proxy model
python train_model.py --is_blackbox False --save_here PM --is_autoencoder 2 >> HPE/$1/log
# # Generate adversarial examples for proxy model
python generate_adversarials.py --model_path PM --adversary_path_x ADX --adversary_path_y ADY --fgsm_eps $1 >> HPE/$1/log
# # Test misclassification accuracy of proxy adversarial examples on blackbox model
python cross_test.py --is_autoencoder 2 --model_path BM --adversary_path_x ADX.npy --adversary_path_y ADY.npy >> HPE/$1/log
python visualize_adex.py --dataset 1

mv BM HPE/$1/
mv PM HPE/$1/
mv ADX.npy HPE/$1/
mv ADXO.npy HPE/$1/
mv ADY.npy HPE/$1/
mv C.pkl HPE/$1/
mv adv_example.png HPE/$1/
mv example.png HPE/$1/
