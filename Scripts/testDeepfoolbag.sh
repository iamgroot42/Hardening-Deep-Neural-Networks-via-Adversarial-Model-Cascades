#!/bin/bash

export TF_CPP_MIN_LOG_LEVEL="2"
dataset=$1 #(cifar100,mnist,svhn)
model=$2 #(path to bag models path)
ppmodel=$3 #(path to proxy model)
method=$4 #(voting/weighted)
iters=50

#Determine set of epsilon values according to dataset
if [ $dataset == "mnist" ]
then
	:
elif [ $dataset == "svhn" ]
then
	:
elif [ $dataset == "cifar100" ]
then
	:
else
	echo "Invalid dataset! Exiting"
	exit
fi

prefix=$(date -d "today" +"%Y%m%d%H%M%S")

python ../Code/deepfool.py --model_path $ppmodel --dataset $dataset --adversary_path_x $prefix"X" --adversary_path_y $prefix"Y" --iters $iters
python ../Code/bagging.py --mode test --dataset $dataset --model_dir $model --data_x $prefix"X.npy" --data_y $prefix"Y.npy"  --predict_mode $method
rm  $prefix"X.npy" $prefix"Y.npy"

