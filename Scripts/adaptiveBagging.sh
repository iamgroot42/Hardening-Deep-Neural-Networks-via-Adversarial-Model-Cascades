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

dataset=$1 # cifar100/mnist/svhn
seedmodel=$2 # path to starting model
bagfolder=$3 # new folder will be made to store bag of models here
cumulative=$4 # cumulative (yes/no)
order=$5 # file containing order of attacks
transfer=$6 #transfer of parameters (yes/no)
seedproxy=$7 # path to starting proxy model

# Make a copy of proxy seed for ending up with new adaptive proxy
temp=$(date -d "today" +"%s")
cp $seedproxy $temp
seedproxy=$temp

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

echo "Seed proxy model will be stored by the name $seedproxy"

COUNTER=1 #Counting models

seeddata=$(date -d "today" +"%s") #Required if cumulative data is to be used
sleep 2 #Make sure name does not clash with prefix inside loop

# Read order of attacks
while read attack
do
	echo "Running attack $COUNTER : $attack"

	prefix=$(date -d "today" +"%s") #Unique per dataset

	# Run attack on proxy
	command="${hashmap[$attack]} --dataset $dataset --adversary_path_x $prefix""X.npy --adversary_path_y $prefix""Y.npy --model_path $seedproxy"
	$command

	# Make copy at proxy's end for finetuning itself
	cp $prefix"X.npy" $prefix"Xproxy.npy"
	cp $prefix"Y.npy" $prefix"Yproxy.npy"

	# Accumulate data
	if [ $cumulative == "yes" ]; then
		if [ "$COUNTER" -gt "1" ]; then
			python coalesce.py $seeddata"X.npy" $seeddata"Y.npy" $prefix"X.npy" $prefix"Y.npy" $seeddata"X.npy" $seeddata"Y.npy"
		else
			mv $prefix"X.npy" $seeddata"X.npy"
			mv $prefix"Y.npy" $seeddata"Y.npy"
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

	lr=0.001
        if [ $dataset == "mnist" ]; then
                lr=0.1
        fi

	# Finetune target model using proxy's attack-data
	python ../Code/bagging.py --learning_rate $lr --nb_epochs 40 --mode finetune --dataset $dataset --seed_model $seeddata"model" --data_x $seeddata"X.npy" --data_y $seeddata"Y.npy" --model_dir $bagfolder

	if [ $cumulative == "no" ]; then
		# Remove temporary data
		rm $seeddata"X.npy" $seeddata"Y.npy"
	fi

	# Update model counter
	COUNTER=$[$COUNTER +1]

	# Add this model to bag
	mv $seeddata"model" $bagfolder/$COUNTER

	# Keras specific change to make sure target model can be loaded in future
	python fix.py $bagfolder/$COUNTER

	# Finetine proxy (make it adapt)
	python ../Code/train_model.py --mode finetune --nb_epochs 20 --learning_rate $lr --save_here $seedproxy --proxy_x $prefix"Xproxy.npy" --proxy_y $prefix"Yproxy.npy" --level blackbox --dataset $dataset

	# Keras specific change to make sure proxy model can be loaded in future
        python fix.py $seedproxy

done < $order