#!/bin/bash
````
python ../Code/generate_adversarials_fgsm.py --model_path $1 --adversary_path_x ADX.npy --adversary_path_y ADY.npy --fgsm_eps 0.000
#python ../Code/cross_test.py --model_path BM_advp --adversary_path_x ADX.npy --adversary_path_y ADY.npy >> FGSMP
#python ../Code/cross_test.py --model_path BM_jsma --adversary_path_x ADX.npy --adversary_path_y ADY.npy >> JSMA
#python ../Code/cross_test.py --model_path BM_jsmap --adversary_path_x ADX.npy --adversary_path_y ADY.npy >> JSMAP
#python ../Code/cross_test.py --model_path BM_scale --adversary_path_x ADX.npy --adversary_path_y ADY.npy >> SCALE
python ../Code/cross_test.py --model_path BM_translate --adversary_path_x ADX.npy --adversary_path_y ADY.npy >> TRANSLATE
#python ../Code/cross_test.py --model_path BM_rotate --adversary_path_x ADX.npy --adversary_path_y ADY.npy >> ROTATE

python ../Code/generate_adversarials_fgsm.py --model_path $1 --adversary_path_x ADX.npy --adversary_path_y ADY.npy --fgsm_eps 0.005
#python ../Code/cross_test.py --model_path BM_advp --adversary_path_x ADX.npy --adversary_path_y ADY.npy >> FGSMP
#python ../Code/cross_test.py --model_path BM_jsma --adversary_path_x ADX.npy --adversary_path_y ADY.npy >> JSMA
#python ../Code/cross_test.py --model_path BM_jsmap --adversary_path_x ADX.npy --adversary_path_y ADY.npy >> JSMAP
#python ../Code/cross_test.py --model_path BM_scale --adversary_path_x ADX.npy --adversary_path_y ADY.npy >> SCALE
python ../Code/cross_test.py --model_path BM_translate --adversary_path_x ADX.npy --adversary_path_y ADY.npy >> TRANSLATE
#python ../Code/cross_test.py --model_path BM_rotate --adversary_path_x ADX.npy --adversary_path_y ADY.npy >> ROTATE

python ../Code/generate_adversarials_fgsm.py --model_path $1 --adversary_path_x ADX.npy --adversary_path_y ADY.npy --fgsm_eps 0.010
#python ../Code/cross_test.py --model_path BM_advp --adversary_path_x ADX.npy --adversary_path_y ADY.npy >> FGSMP
#python ../Code/cross_test.py --model_path BM_jsma --adversary_path_x ADX.npy --adversary_path_y ADY.npy >> JSMA
#python ../Code/cross_test.py --model_path BM_jsmap --adversary_path_x ADX.npy --adversary_path_y ADY.npy >> JSMAP
#python ../Code/cross_test.py --model_path BM_scale --adversary_path_x ADX.npy --adversary_path_y ADY.npy >> SCALE
python ../Code/cross_test.py --model_path BM_translate --adversary_path_x ADX.npy --adversary_path_y ADY.npy >> TRANSLATE
#python ../Code/cross_test.py --model_path BM_rotate --adversary_path_x ADX.npy --adversary_path_y ADY.npy >> ROTATE

python ../Code/generate_adversarials_fgsm.py --model_path $1 --adversary_path_x ADX.npy --adversary_path_y ADY.npy --fgsm_eps 0.015
#python ../Code/cross_test.py --model_path BM_advp --adversary_path_x ADX.npy --adversary_path_y ADY.npy >> FGSMP
#python ../Code/cross_test.py --model_path BM_jsma --adversary_path_x ADX.npy --adversary_path_y ADY.npy >> JSMA
#python ../Code/cross_test.py --model_path BM_jsmap --adversary_path_x ADX.npy --adversary_path_y ADY.npy >> JSMAP
#python ../Code/cross_test.py --model_path BM_scale --adversary_path_x ADX.npy --adversary_path_y ADY.npy >> SCALE
python ../Code/cross_test.py --model_path BM_translate --adversary_path_x ADX.npy --adversary_path_y ADY.npy >> TRANSLATE
#python ../Code/cross_test.py --model_path BM_rotate --adversary_path_x ADX.npy --adversary_path_y ADY.npy >> ROTATE

python ../Code/generate_adversarials_fgsm.py --model_path $1 --adversary_path_x ADX.npy --adversary_path_y ADY.npy --fgsm_eps 0.020
#python ../Code/cross_test.py --model_path BM_advp --adversary_path_x ADX.npy --adversary_path_y ADY.npy >> FGSMP
#python ../Code/cross_test.py --model_path BM_jsma --adversary_path_x ADX.npy --adversary_path_y ADY.npy >> JSMA
#python ../Code/cross_test.py --model_path BM_jsmap --adversary_path_x ADX.npy --adversary_path_y ADY.npy >> JSMAP
#python ../Code/cross_test.py --model_path BM_scale --adversary_path_x ADX.npy --adversary_path_y ADY.npy >> SCALE
python ../Code/cross_test.py --model_path BM_translate --adversary_path_x ADX.npy --adversary_path_y ADY.npy >> TRANSLATE
#python ../Code/cross_test.py --model_path BM_rotate --adversary_path_x ADX.npy --adversary_path_y ADY.npy >> ROTATE

python ../Code/generate_adversarials_fgsm.py --model_path $1 --adversary_path_x ADX.npy --adversary_path_y ADY.npy --fgsm_eps 0.025
#python ../Code/cross_test.py --model_path BM_advp --adversary_path_x ADX.npy --adversary_path_y ADY.npy >> FGSMP
#python ../Code/cross_test.py --model_path BM_jsma --adversary_path_x ADX.npy --adversary_path_y ADY.npy >> JSMA
#python ../Code/cross_test.py --model_path BM_jsmap --adversary_path_x ADX.npy --adversary_path_y ADY.npy >> JSMAP
#python ../Code/cross_test.py --model_path BM_scale --adversary_path_x ADX.npy --adversary_path_y ADY.npy >> SCALE
python ../Code/cross_test.py --model_path BM_translate --adversary_path_x ADX.npy --adversary_path_y ADY.npy >> TRANSLATE
#python ../Code/cross_test.py --model_path BM_rotate --adversary_path_x ADX.npy --adversary_path_y ADY.npy >> ROTATE

python ../Code/generate_adversarials_fgsm.py --model_path $1 --adversary_path_x ADX.npy --adversary_path_y ADY.npy --fgsm_eps 0.030
#python ../Code/cross_test.py --model_path BM_advp --adversary_path_x ADX.npy --adversary_path_y ADY.npy >> FGSMP
#python ../Code/cross_test.py --model_path BM_jsma --adversary_path_x ADX.npy --adversary_path_y ADY.npy >> JSMA
#python ../Code/cross_test.py --model_path BM_jsmap --adversary_path_x ADX.npy --adversary_path_y ADY.npy >> JSMAP
#python ../Code/cross_test.py --model_path BM_scale --adversary_path_x ADX.npy --adversary_path_y ADY.npy >> SCALE
python ../Code/cross_test.py --model_path BM_translate --adversary_path_x ADX.npy --adversary_path_y ADY.npy >> TRANSLATE
#python ../Code/cross_test.py --model_path BM_rotate --adversary_path_x ADX.npy --adversary_path_y ADY.npy >> ROTATE
