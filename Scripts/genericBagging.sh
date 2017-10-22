#!/bin/bash

export TF_CPP_MIN_LOG_LEVEL="2"

declare -A hashmap

#FGSM
fgsm_eps=0.03
#Elastic
elastic_gamma=1e-3
#Deepfool
iters=50
#Virtual
num_iters=1
xi=1e-6
eps=2.0
#Madry
madry_eps=0.03
#JSMA
jsma_gamma=0.1
theta=1.0
n_subset_classes=10

dataset=$1 #(cifar100,mnist,svhn)
seedmodel=$2 #(path to starting model)
bagfolder=$3 #(new folder will be made to store bag of models here)
cumulative=$4 #(yes/no)
order=$5 #(file containing order of attacks)
transfer=$6 #transfer of parameters (yes/no)

hashmap["fgsm"]="python ../Code/fgsm.py --fgsm_eps $fgsm_eps "
hashmap["jsma"]="python ../Code/jsma.py --gamma $jsma_gamma --theta $theta --n_subset_classes $n_subset_classes "
hashmap["elastic"]="python ../Code/elastic.py --gamma $elastic_gamma "
hashmap["carlini"]="python ../Code/elastic.py --gamma 0 "
hashmap["deepfool"]="python ../Code/deepfool.py --iters $iters "
hashmap["madry"]="python ../Code/madry.py --epsilon $madry_eps "
hashmap["virtual"]="python ../Code/virtual.py --num_iters $num_iters --xi $xi --eps $eps "

if [ $dataset == "mnist" ]
        then
                fgsm_eps=0.1
		madry_eps=0.1
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

mkdir -p $bagfolder

# Copy initial seed model to directory
python fix.py $seedmodel
cp $seedmodel $bagfolder/1

COUNTER=1 #Counting models
seeddata=$(date -d "today" +"%s") #Required if cumulative data is to be used
sleep 2 #Make sure name does not clash with prefix inside loop

# Read order of attacks
while read attack
do
	prefix=$(date -d "today" +"%s") #Unique per dataset

	#Run attack
	command="${hashmap[$attack]} --dataset $dataset --adversary_path_x $prefix""X.npy --adversary_path_y $prefix""Y.npy --model_path $seedmodel"
	$command

	# Accumulate data
	if [ $cumulative == "yes" ]; then
		if [ "$COUNTER" -gt "1" ]; then
			python coalesce.py $seeddata"X.npy" $seeddata"Y.npy" $prefix"X.npy" $prefox"Y.npy" $seeddata"X.npy" $seeddata"Y.npy"
		else
			cp $prefix"X.npy" $seeddata"X.npy"
			cp $prefix"Y.npy" $seeddata"Y.npy"
		fi
	else
		mv $prefix"X.npy" $seeddata"X.npy"
		mv $prefix"Y.npy" $seeddata"Y.npy"
	fi

	selectedmodel=$seedmodel
	# Pick model according to desired option
	if [ $transfer == "yes" ]; then
		selectedmodel=$bagfolder/$COUNTER
	fi

	# Make a copy of selected model for finetuning
	cp $selectedmodel $seeddata"model"

	# Finetune data
	python ../Code/bagging.py --nb_epochs 10 --mode finetune --dataset $dataset --seed_model $seeddata"model" --data_x $seeddata"X.npy" --data_y $seeddata"Y.npy" --model_dir $bagfolder

	# Remove temporary data
	rm $seeddata"X.npy" $seeddata"Y.npy"

	# Update model counter
	COUNTER=$[$COUNTER +1]

	# Add this model to bag
	mv $seeddata"model" $bagfolder/$COUNTER

	#Keras specific change to make sure model can be loaded in future
	python fix.py $bagfolder/$COUNTER

done < $order
