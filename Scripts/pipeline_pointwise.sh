#!/bin/bash


python ../Code/pixelwise_gen.py --model_path BM --adversary_path_x ADX.npy --adversary_path_y ADY.npy
# Sample an adversarial image for visualization
# python ../Code/visualize_adex.py --dataset 1
