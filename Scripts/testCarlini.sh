#!/bin/bash

export TF_CPP_MIN_LOG_LEVEL="2"
dataset=$1 #(cifar100,mnist,svhn)
model=$2 #(path to blackbox model)
ppmodel=$3 #(path to proxy model)

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

python ../Code/carlini.py --model_path $ppmodel --dataset $dataset --data_x $prefix"X" --data_y $prefix"Y"
python ../Code/cross_test.py --model_path $model --data_x $prefix"X.npy" --data_y $prefix"Y.npy" --dataset $dataset --proxy_data False
rm  $prefix"X.npy" $prefix"Y.npy"
echo $prefix
