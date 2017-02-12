#!/bin/bash

epsilon=$1
perclass=$2

mkdir -p HPE/$1
# Train blackbox model
python train_model.py --is_blackbox True --save_here BM --is_autoencoder 2 --num_clusters 64 >> HPE/$epsilon/log
# Generate training data for proxy network
python cross_test.py --model_path BM  --is_autoencoder 2 --proxy_data True --per_class_adv $perclass --is_autoencoder 3 >> HPE/$epsilon/log
# Train proxy model
python train_model.py --is_blackbox False --save_here PM --is_autoencoder 2 >> HPE/$epsilon/log
# # Generate adversarial examples for proxy model
python generate_adversarials.py --model_path PM --adversary_path_x ADX --adversary_path_y ADY --fgsm_eps $epsilon >> HPE/$epsilon/log
# # Test misclassification accuracy of proxy adversarial examples on blackbox model
python cross_test.py --is_autoencoder 2 --model_path BM --adversary_path_x ADX.npy --adversary_path_y ADY.npy >> HPE/$epsilon/log
python visualize_adex.py --dataset 1

mv BM HPE/$epsilon/
mv PM HPE/$epsilon/
mv ADX.npy HPE/$epsilon/
mv ADXO.npy HPE/$epsilon/
mv ADY.npy HPE/$epsilon/
mv PX.npy HPE/$epsilon/
mv PY.npy HPE/$epsilon/
mv C.pkl HPE/$epsilon/
mv adv_example.png HPE/$epsilon/
mv example.png HPE/$epsilon/
