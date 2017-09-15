#!/bin/bash

if [ ! -z $5 ]; then
	echo "Generate union, intersection models (with scale)"
	python ../Code/mixture_attack_scale.py --models_directory $1 --fgsm_eps $3 --models_data_directory $2
else
	echo "Generate union, intersection models"
	python ../Code/mixture_attack.py --models_directory $1 --fgsm_eps $3 --models_data_directory $2
fi

mv $2labels_union.npy labels_union.npy
mv $2labels_intersection.npy labels_intersection.npy

python coalesce.py $2union.npy labels_union.npy $2intersection.npy labels_intersection.npy potatox potatoy

python ../Code/cross_test.py --model_path $4 --adversary_path_x potatox.npy --adversary_path_y potatoy.npy

mv potatox.npy PX.npy
mv potatoy.npy PY.npy

cp $4 BM
python ../Code/train_model.py --is_blackbox True --save_here BM --specialCNN sota --retraining True --batch_size 16 --learning_rate 0.0001 --nb_epochs 40
mv BM BM_rotate
