#!/bin/bash

perclass=$1

mkdir -p ../Data/Pixelwise
# Train blackbox model
python ../Code/train_model.py --is_blackbox True --save_here BM --specialCNN sota --batch_size 16 --learning_rate 0.05 --nb_epochs 200 >> ../Data/Pixelwise/log
# Generate adversarial examples for proxy model
python ../Code/pixelwise_gen.py --model_path BM --adversary_path_x ADX.npy --adversary_path_y ADY.npy >> ../Data/Pixelwise/log
# Test misclassification accuracy of proxy adversarial examples on blackbox model
python ../Code/cross_test.py --model_path BM --adversary_path_x ADX.npy --adversary_path_y ADY.npy >> ../Data/Pixelwise/log
# Sample an adversarial image for visualization
python ../Code/visualize_adex.py --dataset 1

#mv BM ../Data/Pixelwise/
##mv ADX.npy ../Data/Pixelwise/
#mv ADXO.npy ../Data/Pixelwise/
#mv ADY.npy ../Data/Pixelwise/
#mv adv_example.png ../Data/Pixelwise/
#mv example.png ../Data/Pixelwise/

