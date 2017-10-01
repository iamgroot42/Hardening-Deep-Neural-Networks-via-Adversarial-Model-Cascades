#!/bin/bash

export TF_CPP_MIN_LOG_LEVEL="2"
dataset=$1 #(cifar100,mnist,svhn)
model=$2 #(path to blackbox model)
ppc=$3 #(proxy examples per class)
label_smooth=0.0

#Determine set of epsilon values according to dataset
if [ $dataset == "mnist" ]
	then
  		declare -a epsilon_values=(0 0.04 0.06 0.08 0.10)
elif [ $dataset == "svhn" ]
	then
		declare -a epsilon_values=(0 10 25 50 100)
elif [ $dataset == "cifar100" ]
	then
		declare -a epsilon_values=(0 0.005 0.010 0.015 0.020 0.025 0.030)
else
	echo "Invalid dataset! Exiting"
	exit
fi

for epsilon in "${epsilon_values[@]}"
do

	prefix=$(date -d "today" +"%Y%m%d%H%M%S")

	python ../Code/cross_test.py --model_path $model --proxy_x $prefix"X" --proxy_y $prefix"Y" --per_class_adv $ppc --dataset $dataset --proxy_data True
	python ../Code/train_model.py --dataset $dataset --nb_epochs 100 --save_here $prefix --level proxy --proxy_x $prefix"X.npy" --proxy_y $prefix"Y.npy"  --label_smooth $label_smooth
	python ../Code/generate_adversarials_fgsm.py --fgsm_eps $epsilon --model_path $prefix --dataset $dataset --adversary_path_x $prefix"X" --adversary_path_y $prefix"Y"
	python ../Code/cross_test.py --model_path $model --adversary_path_x $prefix"X.npy" --adversary_path_y $prefix"Y.npy" --per_class_adv $ppc --dataset $dataset --proxy_data False
	rm $prefix $prefix"X.npy" $prefix"Y.npy"
done

