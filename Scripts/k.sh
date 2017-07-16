#!/bin/bash

#echo "0.000" >> LOG
#python ../Code/generate_adversarials.py --model_path PM --adversary_path_x ADX.npy --adversary_path_y ADY.npy --fgsm_eps 0.000 >> LOG
#python ../Code/cross_test.py --model_path BM --adversary_path_x ADX.npy --adversary_path_y ADY.npy --is_autoencoder 3 >> LOG

#echo "0.005" >> LOG
#python ../Code/generate_adversarials.py --model_path PM --adversary_path_x ADX.npy --adversary_path_y ADY.npy --fgsm_eps 0.005 >> LOG
#python ../Code/cross_test.py --model_path BM --adversary_path_x ADX.npy --adversary_path_y ADY.npy --is_autoencoder 3 >> LOG

#echo "0.010" >> LOG
#python ../Code/generate_adversarials.py --model_path PM --adversary_path_x ADX.npy --adversary_path_y ADY.npy --fgsm_eps 0.010 >> LOG
#python ../Code/cross_test.py --model_path BM --adversary_path_x ADX.npy --adversary_path_y ADY.npy --is_autoencoder 3 >> LOG

#echo "0.015" >> LOG
#python ../Code/generate_adversarials.py --model_path PM --adversary_path_x ADX.npy --adversary_path_y ADY.npy --fgsm_eps 0.015 >> LOG
#python ../Code/cross_test.py --model_path BM --adversary_path_x ADX.npy --adversary_path_y ADY.npy --is_autoencoder 3 >> LOG

#echo "0.020" >> LOG
#python ../Code/generate_adversarials.py --model_path PM --adversary_path_x ADX.npy --adversary_path_y ADY.npy --fgsm_eps 0.020 >> LOG
#python ../Code/cross_test.py --model_path BM --adversary_path_x ADX.npy --adversary_path_y ADY.npy --is_autoencoder 3 >> LOG

#echo "0.025" >> LOG
#python ../Code/generate_adversarials.py --model_path PM --adversary_path_x ADX.npy --adversary_path_y ADY.npy --fgsm_eps 0.025 >> LOG
#python ../Code/cross_test.py --model_path BM --adversary_path_x ADX.npy --adversary_path_y ADY.npy --is_autoencoder 3 >> LOG

echo "0.030" >> LOG
python ../Code/generate_adversarials.py --model_path PM --adversary_path_x ADX.npy --adversary_path_y ADY.npy --fgsm_eps 0.030  >> LOG
python ../Code/cross_test.py --model_path BM --adversary_path_x ADX.npy --adversary_path_y ADY.npy --is_autoencoder 3 >> LOG


