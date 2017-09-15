#!/bin/bash

python ../Code/generate_adversarials_fgsm.py --model_path $1 --adversary_path_x ADX.npy --adversary_path_y ADY.npy --fgsm_eps 0.005
python ../Code/generate_adversarials_fgsm.py --model_path $1 --adversary_path_x ADX.npy --adversary_path_y ADY.npy --fgsm_eps 0.010
python ../Code/generate_adversarials_fgsm.py --model_path $1 --adversary_path_x ADX.npy --adversary_path_y ADY.npy --fgsm_eps 0.015
python ../Code/generate_adversarials_fgsm.py --model_path $1 --adversary_path_x ADX.npy --adversary_path_y ADY.npy --fgsm_eps 0.020
python ../Code/generate_adversarials_fgsm.py --model_path $1 --adversary_path_x ADX.npy --adversary_path_y ADY.npy --fgsm_eps 0.025
python ../Code/generate_adversarials_fgsm.py --model_path $1 --adversary_path_x ADX.npy --adversary_path_y ADY.npy --fgsm_eps 0.030

