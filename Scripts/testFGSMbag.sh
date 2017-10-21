#!/bin/bash

export TF_CPP_MIN_LOG_LEVEL="2"
dataset=$1 #(cifar100,mnist,svhn)
model=$2 #(path to directory of bag of models)
ppmodel=$3 #(path to proxy model)

#Determine set of epsilon values according to dataset
if [ $dataset == "mnist" ]
	then
  		declare -a epsilon_values=(0 0.04 0.06 0.08 0.10 0.25)
elif [ $dataset == "svhn" ]
	then
		declare -a epsilon_values=(0 0.007 0.015 0.03 0.06)
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

	python ../Code/fgsm.py --fgsm_eps $epsilon --model_path $ppmodel --dataset $dataset --adversary_path_x $prefix"X" --adversary_path_y $prefix"Y"
	python ../Code/bagging.py --input_model_dir $model --data_x $prefix"X.npy" --data_y $prefix"Y.npy" --mode test
	rm  $prefix"X.npy" $prefix"Y.npy"
	echo "Results above for " $epsilon
done
