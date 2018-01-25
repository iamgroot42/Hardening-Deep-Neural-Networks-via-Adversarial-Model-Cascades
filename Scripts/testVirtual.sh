#!/bin/bash

export TF_CPP_MIN_LOG_LEVEL="2"
dataset=$1 #(cifar100,mnist,svhn)
model=$2 #(path to blackbox model)
ppmodel=$3 #(path to proxy model)
num_iters=1
xi=1e-6
eps=2.0

#Determine set of epsilon values according to dataset
if [ $dataset == "mnist" ]
then
	:
elif [ $dataset == "svhn" ]
then
	:
elif [ $dataset == "cifar10" ]
then
	:
else
	echo "Invalid dataset! Exiting"
	exit
fi

prefix=$(date -d "today" +"%Y%m%d%H%M%S")

python ../Code/virtual.py --model_path $ppmodel --dataset $dataset --data_x $prefix"X" --data_y $prefix"Y" --num_iters $num_iters --xi $xi --eps $eps
python ../Code/cross_test.py --model_path $model --data_x $prefix"X.npy" --data_y $prefix"Y.npy" --dataset $dataset
rm  $prefix"X.npy" $prefix"Y.npy"
echo $prefix
