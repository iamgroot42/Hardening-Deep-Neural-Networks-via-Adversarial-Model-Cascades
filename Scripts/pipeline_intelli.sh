#!/bin/bash

epsilon=$1
perclass=$2
bb_model=NEW_BMADV$2
p_model=$4

cp Models/Blackbox/CNN/BM_5260 $bb_model
mkdir -p IntelliCNN$perclass/$epsilon
# Generate adversarial examples fo refining network strength
python ../Code/generate_adversarials_fgsm.py --model_path $p_model --adversary_path_x PX$perclass.npy --adversary_path_y PY$perclass.npy --fgsm_eps $epsilon >> IntelliCNN$perclass/$epsilon/log
# Finetune blackbox model using adversarial data
python ../Code/train_model.py --is_blackbox True --adversary_path_x PX$perclass --adversary_path_y PY$perclass --save_here $bb_model --specialCNN sota --retraining True --batch_size 16 --learning_rate 0.0001 --nb_epochs 70 >> IntelliCNN$perclass/$epsilon/log
# Fix generated model
python fix.py $bb_model
# Generate training data for proxy network
python ../Code/cross_test.py --model_path $bb_model --proxy_data True --per_class_adv $perclass >> IntelliCNN$perclass/$epsilon/log
# Train proxy model
python ../Code/train_model.py --is_blackbox False --save_here PM --batch_size 16 --learning_rate 0.005 --nb_epochs 200 >> IntelliCNN$perclass/$epsilon/log
# Generate adversarial examples for proxy model
python ../Code/generate_adversarials_fgsm.py --model_path PM --adversary_path_x ADX$perclass.npy --adversary_path_y ADY$perclass.npy --fgsm_eps $epsilon >> IntelliCNN$perclass/$epsilon/log
# Test misclassification accuracy of proxy adversarial examples on blackbox model
python ../Code/cross_test.py --model_path $bb_model --adversary_path_x ADX$perclass.npy --adversary_path_y ADY$perclass.npy >> IntelliCNN$perclass/$epsilon/log
# Sample an adversarial image for visualization
python ../Code/visualize_adex.py --dataset 1
