#!/bin/bash

# No transfer of parameters, keep increasing bag (blackbox examples)

dataset=$1 #(cifar100,mnist,svhn)
epsilon=$2 #value of epsilon
cumulative=$3 #(yes/no)

fgsm_x=$4
fgsm_x=$5

jsma_x=$6
jsma_x=$7


temporary_folder=$(date -d "today" +"%Y%m%d%H%M%S")
bag_dir="BAG_SETUP1"dataset
mkdir $bag_dir $temporary_folder

# Train a CNN, test how well it worked so far
python ../Code/bagging.py --mode train --dataset $dataset --input_model_dir $temporary_folder --add_model True --output_model_dir $temporary_folder
python fix.py $temporary_folder/1
cp $temporary_folder/1 $bag_dir/1

# Finetune using Blackbox FGSM Noise, test how well it worked so far
python ../Code/bagging.py --mode finetune --dataset $dataset --input_model_dir $temporary_folder --add_model False --output_model_dir $temporary_folder --data_x $fgsm_x --data_y $fgsm_y
python fix.py $temporary_folder/1
cp $temporary_folder/1 $bag_dir/2
cp $bag_dir/1 $temporary_folder/1

# Finetune using Blackbox JSMA Noise, test how well it worked so far
python ../Code/bagging.py --mode finetune --dataset $dataset --input_model_dir $temporary_folder --add_model False --output_model_dir $temporary_folder --data_x $jsma_x --data_y $jsma_y
python fix.py $temporary_folder/1
cp $temporary_folder/1 $bag_dir/3

#Clean up
rm -r $temporary_folder
rm  -f $temporary_folder"adx.npy"  $temporary_folder"ady.npy"
rm  -f $temporary_folder"ad2x.npy"  $temporary_folder"ad2y.npy"
