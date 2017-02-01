#!/bin/bash

epsilon=$1
mkdir -p CNNSVM/$epsilon
# Train blackbox model
python train_model.py --is_blackbox True --save_here BM --is_autoencoder 3 >> CNNSVM/$epsilon/log
# Train proxy model
python train_model.py --is_blackbox False --save_here PM --is_autoencoder 3 >> CNNSVM/$epsilon/log
# Generate adversarial examples for proxy model
python generate_adversarials.py --model_path PM --adversary_path_x ADX --adversary_path_y ADY #--fgsm_eps $epsilon >> CNNSVM/$epsilon/log
# Test misclassification accuracy of proxy adversarial examples on blackbox model
python cross_test.py --model_path BM --adversary_path_x ADX.npy --adversary_path_y ADY.npy --is_autoencoder 3 >> CNNSVM/$epsilon/log
# Sample an adversarial image for visualization
# python visualize_adex.py --dataset 1

mv BM CNNSVM/$epsilon/
mv PM CNNSVM/$epsilon/
mv ADX.npy CNNSVM/$epsilon/
mv ADY.npy CNNSVM/$epsilon/
mv C.pkl CNNSVM/$epsilon/
# mv adv_example.png CNNSVM/$epsilon/
