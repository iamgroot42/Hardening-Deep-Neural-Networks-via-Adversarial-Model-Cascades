#!/bin/bash

K=$2

mkdir -p HPE/$K
# Train blackbox model
python train_model.py --is_blackbox True --save_here BM --is_autoencoder 2 >> HPE/$K/log
echo "blackbox"
# Train proxy model
python train_model.py --is_blackbox False --save_here PM --is_autoencoder 2 >> HPE/$K/log
echo "proxy"
# # Generate adversarial examples for proxy model
python generate_adversarials.py --model_path PM --adversary_path_x ADX --adversary_path_y ADY --fgsm_eps 0.10 >> HPE/$K/log
echo "advgen"
# # Test misclassification accuracy of proxy adversarial examples on blackbox model
python cross_test.py --is_autoencoder 2 --model_path BM --adversary_path_x ADX.npy --adversary_path_y ADY.npy >> HPE/$K/log
echo "test_acc"
# # Sample an adversarial image for visualization
# python visualize_adex.py --dataset 1

mv BM HPE/$K/
mv PM HPE/$K/
mv ADX.npy HPE/$K/
mv ADY.npy HPE/$K/
mv C.pkl HPE/$K/
#mv adv_example.png HPE/$K/
