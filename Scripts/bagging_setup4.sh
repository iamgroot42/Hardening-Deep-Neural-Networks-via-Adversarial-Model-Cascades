#!/bin/bash

# No transfer of parameters, keep increasing bag (blackbox examples)

export TF_CPP_MIN_LOG_LEVEL="2"
dataset=$1 #(cifar100,mnist,svhn)
epsilon=$2 #value of epsilon
cumulative=$3 #(yes/no)

fgsm_x=$4
fgsm_y=$5

jsma_x=$6
jsma_y=$7


temporary_folder=$(date -d "today" +"%Y%m%d%H%M%S")
bag_dir=$3"BAG_SETUP2"$dataset
mkdir $bag_dir $temporary_folder

# Train a CNN, test how well it worked so far
python ../Code/bagging.py --mode train --dataset $dataset --input_model_dir $temporary_folder --add_model True --output_model_dir $temporary_folder

python fix.py $temporary_folder/1
cp $temporary_folder/1 $bag_dir/1

# Finetune using Blackbox FGSM Noise, test how well it worked so far
python ../Code/bagging.py --mode finetune --dataset $dataset --input_model_dir $temporary_folder --add_model False --output_model_dir $temporary_folder --data_x $fgsm_x --data_y $fgsm_y
python fix.py $temporary_folder/1
cp $temporary_folder/1 $bag_dir/2
cp $bag_dir/2 $temporary_folder/1

# Finetune using Blackbox JSMA Noise, test how well it worked so far
if [ $cumulative == "yes" ]; then
	python coalesce.py $jsma_x $jsma_y $fgsm_x $fgsm_y $bag_dir/"tempux" $bag_dir/"tempuy"
	python ../Code/bagging.py --mode finetune --dataset $dataset --input_model_dir $temporary_folder --add_model False --output_model_dir $temporary_folder --data_x $bag_dir/"tempux.npy" --data_y $bag_dir/"tempuy.npy"
	rm $bag_dir/"tempux.npy" $bag_dir/"tempuy.npy"
else
	python ../Code/bagging.py --mode finetune --dataset $dataset --input_model_dir $temporary_folder --add_model False --output_model_dir $temporary_folder --data_x $jsma_x --data_y $jsma_y
fi

python fix.py $temporary_folder/1
cp $temporary_folder/1 $bag_dir/3

#Clean up
rm -r $temporary_folder
