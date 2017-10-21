#!/bin/bash

export TF_CPP_MIN_LOG_LEVEL="2"

declare -A hashmap

#FGSM
fgsm_eps=0.0
#Elastic
elastic_gamma=1e-3
#Deepfool
iters=50
#Virtual
num_iters=1
xi=1e-6
eps=2.0
#Madry
madry_eps=0.0
#JSMA
jsma_gamma=0.1
theta=1.0
n_subset_classes=10

dataset=$1 #(cifar100,mnist,svhn)
seedmodel=$2 #(path to starting model)
bagfolder=$3 #(new folder will be made to store bag of models here)
cumulative=$4 #(yes/no)
order=$5 #(file containing order of attacks)
transfer=$6 #transfer of parameters

hashmap["fgsm"]="python ../Code/fgsm.py --fgsm_eps $fgsm_eps "
hashmap["jsma"]="python ../Code/jsma.py --gamma $jsma_gamma --theta $theta --n_subset_classes $n_subset_classes "
hashmap["elastic"]="python ../Code/elastic.py --gamma $elastic_gamma "
hashmap["carlini"]="python ../Code/elastic.py --gamma 0 "
hashmap["deepfool"]="python ../Code/deepfool.py --iters $iters "
hashmap["madry"]="python ../Code/madry.py --epsilon $madry_eps "
hashmap["virtual"]="python ../Code/virtual.py --num_iters $num_iters --xi $xi --eps $eps "

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

mkdir -p $bagfolder
cp $seedmodel $bagfolder/1

# Copy initial seed model to directory
python fix.py $bagfolder/1

COUNTER=1 #Counting models
seeddata=$(date -d "today" +"%s") #Required if cumulative data is to be used
sleep 2 #Make sure name does not clash with prefix inside loop


# Read order of attacks
while read attack
do
	prefix=$(date -d "today" +"%s") #Unique per dataset

	#Run attack
	"${hashmap[$attack]} --dataset $dataset --adversary_path_x $prefix""X.npy --adversary_path_y $prefix""Y.npy"

	# Accumulate data
	if [ $cumulative == "yes" "$COUNTER" -gt "1" ]; then
		if [ "$COUNTER" -gt "1" ]; then
			python coalesce.py $seeddata"X.npy" $seeddata"Y.npy" $prefix"X.npy" $prefox"Y.npy" $seeddata"X.npy" $seeddata"Y.npy"
		else
			cp $prefix"X.npy" $seeddata"X.npy"
			cp $prefix"Y.npy" $seeddata"Y.npy"
		fi
	fi

	selectedmodel=$seedmodel
	# Pick model according to desired option
	if [ $transfer == "yes" ]; then
		selectedmodel=$COUNTER
	fi

	# Finetune data
	python ../Code/bagging.py --mode finetune --dataset $dataset --seed_model $seedmodel --output_model_dir $bagfolder --data_x $seeddata"X.npy" --data_y $seeddata"Y.npy"

	# Remove temporary data
	rm $prefix"X.npy" $prefix"Y.npy"

	# Update model counter
	COUNTER=$[$COUNTER +1]

	#Keras specific change to make sure model can be loaded in future
	python fix.py $temporary_folder/1

done < $4
