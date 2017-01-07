#!/bin/bash

# Train blackbox model
python train_model.py --is_blackbox True --save_here BM
# Train proxy model
python train_model.py --is_blackbox False --save_here PM
# Generate adversarial examples for proxy model
python generate_adversarials.py --model_path PM --adversary_path_x ADX --adversary_path_y ADY --fgsm_eps 0.8
# Test misclassification accuracy of proxy adversarial examples on blackbox model
python cross_test.py --model_path BM --adversary_path_x ADX.npy --adversary_path_y ADY.npy

