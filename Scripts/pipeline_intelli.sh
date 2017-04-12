#!/bin/bash

epsilon=$1
perclass=$2

mkdir -p ../Data/IntelliCNN/$epsilon
# Train blackbox model
python ../Code/train_model.py --is_blackbox True --save_here BM --specialCNN sota --batch_size 16 --learning_rate 0.0001 --nb_epochs 300 >> ../Data/IntelliCNN/$epsilon/log
# Generate adversarial examples fo refining network strength
python ../Code/generate_adversarials.py --model_path BM --adversary_path_x PX.npy --adversary_path_y PY.npy --fgsm_eps $epsilon >> ../Data/IntelliCNN/$epsilon/log
# Remove previous model
rm BM
# Finetune blackbox model using adversarial data
python ../Code/train_model.py --is_blackbox True --save_here BM --specialCNN sota --retraining True >> ../Data/IntelliCNN/$epsilon/log
# Generate training data for proxy network
python ../Code/cross_test.py --model_path BM --proxy_data True --per_class_adv $perclass >> ../Data/IntelliCNN/$epsilon/log
# Train proxy model
python ../Code/train_model.py --is_blackbox False --save_here PM  --batch_size 16 --learning_rate 0.01 --nb_epochs 500 >> ../Data/IntelliCNN/$epsilon/log
# Generate adversarial examples for proxy model
python ../Code/generate_adversarials.py --model_path PM --adversary_path_x ADX.npy --adversary_path_y ADY.npy --fgsm_eps $epsilon >> ../Data/IntelliCNN/$epsilon/log
# Test misclassification accuracy of proxy adversarial examples on blackbox model
python ../Code/cross_test.py --model_path BM --adversary_path_x ADX.npy --adversary_path_y ADY.npy >> ../Data/IntelliCNN/$epsilon/log
# Sample an adversarial image for visualization
python ../Code/visualize_adex.py --dataset 1

mv BM ../Data/IntelliCNN/$epsilon/
mv PM ../Data/IntelliCNN/$epsilon/
mv ADX.npy ../Data/IntelliCNN/$epsilon/
mv ADXO.npy ../Data/IntelliCNN/$epsilon/
mv ADY.npy ../Data/IntelliCNN/$epsilon/
mv PX.npy ../Data/IntelliCNN/$epsilon/
mv PY.npy ../Data/IntelliCNN/$epsilon/
mv adv_example.png ../Data/IntelliCNN/$epsilon/
mv example.png ../Data/IntelliCNN/$epsilon/
