#!/bin/bash

# No transfer of parameters, keep increasing bag (whitebox examples)

export TF_CPP_MIN_LOG_LEVEL="2"
dataset=$1 #(cifar100,mnist,svhn)
epsilon=$2 #value of epsilon
cumulative=$3 #(yes/no)

temporary_folder=$(date -d "today" +"%Y%m%d%H%M%S")
bag_dir=$3"BAG_SETUP3"$dataset
mkdir $bag_dir $temporary_folder

# Train a CNN, test how well it worked so far
python ../Code/bagging.py --mode train --dataset $dataset --input_model_dir $temporary_folder --add_model True --output_model_dir $temporary_folder
python fix.py $temporary_folder/1
cp $temporary_folder/1 $bag_dir/1

# Finetune using Whitebox FGSM Noise, test how well it worked so far
python ../Code/generate_adversarials_fgsm.py --dataset $dataset --model_path $temporary_folder/1 --fgsm_eps $2 --adversary_path_x $temporary_folder"adx" --adversary_path_y $temporary_folder"ady"
python ../Code/bagging.py --mode finetune --dataset $dataset --input_model_dir $temporary_folder --add_model False --output_model_dir $temporary_folder --data_x $temporary_folder"adx.npy" --data_y $temporary_folder"ady.npy"
python fix.py $temporary_folder/1
cp $temporary_folder/1 $bag_dir/2
cp $bag_dir/2 $temporary_folder/1

# Finetune using Whitebox JSMA Noise, test how well it worked so far
python ../Code/fool_jsma.py --dataset $dataset --n_subset_classes 10 --model_path $temporary_folder/1 --adversary_path_x $temporary_folder"ad2x" --adversary_path_y $temporary_folder"ad2y"
if [$cumulative == "yes"]; then
	python coalesce.py $temporary_folder"adx.npy" $temporary_folder"ady.npy" $temporary_folder"ad2x.npy" $temporary_folder"ad2y.npy" $temporary_folder"adx.npy" $temporary_folder"ady.npy" 
else
	mv $temporary_folder"ad2x.npy" $temporary_folder"adx.npy"
	mv $temporary_folder"ad2y.npy" $temporary_folder"ady.npy"
fi
python ../Code/bagging.py --mode finetune --dataset $dataset --input_model_dir $temporary_folder --add_model False --output_model_dir $temporary_folder --data_x $temporary_folder"adx.npy" --data_y $temporary_folder"ady.npy"
python fix.py $temporary_folder/1
cp $temporary_folder/1 $bag_dir/3

#Clean up
rm -r $temporary_folder
rm  -f $temporary_folder"adx.npy"  $temporary_folder"ady.npy"
rm  -f $temporary_folder"ad2x.npy"  $temporary_folder"ad2y.npy"
