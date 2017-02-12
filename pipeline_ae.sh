#!/bin/bash

epsilon=$1
mkdir -p AdvOutputs/$epsilon
# Train blackbox model
python train_model.py --is_blackbox True --save_here BM --is_autoencoder 1 >> AdvOutputs/$epsilon/log
# Train proxy model
python train_model.py --is_blackbox False --save_here PM --is_autoencoder 1 >> AdvOutputs/$epsilon/log
# Generate adversarial examples for proxy model
python generate_adversarials.py --model_path PM --adversary_path_x ADX --adversary_path_y ADY --fgsm_eps $epsilon >> AdvOutputs/$epsilon/log
# Test misclassification accuracy of proxy adversarial examples on blackbox model
python cross_test.py --model_path BM --adversary_path_x ADX.npy --adversary_path_y ADY.npy >> AdvOutputs/$epsilon/log
# Sample an adversarial image for visualization
#python visualize_adex.py --dataset 1

mv BM AdvOutputs/$epsilon/
mv PM AdvOutputs/$epsilon/
mv ADX.npy AdvOutputs/$epsilon/
mv ADXO.npy AdvOutputs/$epsilon/
mv ADY.npy AdvOutputs/$epsilon/
mv adv_example.png AdvOutputs/$epsilon/
mv example.png AdvOutputs/$epsilon/
