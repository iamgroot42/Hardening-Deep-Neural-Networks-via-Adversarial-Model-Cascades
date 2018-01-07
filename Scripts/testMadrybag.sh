#!/bin/bash

export TF_CPP_MIN_LOG_LEVEL="2"
dataset=$1 #(cifar100,mnist,svhn)
model=$2 #(path to blackbox model)
ppmodel=$3 #(path to proxy model)

#Determine set of epsilon values according to dataset
if [ $dataset == "mnist" ]
	then
  		declare -a epsilon_values=(0 0.04 0.06 0.08 0.10 0.25)
elif [ $dataset == "svhn" ]
	then
		declare -a epsilon_values=(0 0.007 0.015 0.03 0.6)
elif [ $dataset == "cifar10" ]
	then
		declare -a epsilon_values=(0 0.007 0.015 0.03 0.6)
else
	echo "Invalid dataset! Exiting"
	exit
fi

for epsilon in "${epsilon_values[@]}"
do

	prefix=$(date -d "today" +"%Y%m%d%H%M%S")
	echo "For epsilon $epsilon:"
	python ../Code/madry.py --epsilon $epsilon --model_path $ppmodel --dataset $dataset --data_x $prefix"X" --data_y $prefix"Y"
	python ../Code/bagging.py --mode test --dataset $dataset --model_dir $model --data_x $prefix"X.npy" --data_y $prefix"Y.npy" --predict_mode weighted
	rm  $prefix"X.npy" $prefix"Y.npy"
done
