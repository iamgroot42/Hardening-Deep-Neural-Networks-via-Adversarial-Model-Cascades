#!/bin/bash

export TF_CPP_MIN_LOG_LEVEL="2"

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
	command="python ../Code/attack.py --mode harden --attack_name $attack --dataset $dataset --save_here $prefix --model_path $seedproxy"
	$command

	# Make copy at proxy's end for finetuning itself
	cp $prefix"_x.npy" $prefix"_xproxy.npy"
	cp $prefix"_y.npy" $prefix"_yproxy.npy"

	# Accumulate data
	if [ $cumulative == "yes" ]; then
		if [ "$COUNTER" -gt "1" ]; then
			python coalesce.py $seeddata"_x.npy" $seeddata"_y.npy" $prefix"_x.npy" $prefix"_y.npy" $seeddata"_x.npy" $seeddata"_y.npy"
		else
			mv $prefix"_x.npy" $seeddata"_x.npy"
			mv $prefix"_y.npy" $seeddata"_y.npy"
		fi
	else
		mv $prefix"_x.npy" $seeddata"_x.npy"
		mv $prefix"_y.npy" $seeddata"_y.npy"
	fi

	selectedmodel=$seedmodel
	# Pick model according to desired option
	if [ $transfer == "yes" ]; then
		selectedmodel=$bagfolder/$COUNTER
	fi

	# Make a copy of selected model for finetuning
	cp $selectedmodel $seeddata"model"

	lr=1

	# Finetune target model using proxy's attack-data
	python ../Code/bagging.py --learning_rate $lr --nb_epochs 150 --mode finetune --dataset $dataset --seed_model $seeddata"model" --data_x $seeddata"_x.npy" --data_y $seeddata"_y.npy" --model_dir $bagfolder

	if [ $cumulative == "no" ]; then
		# Remove temporary data
		rm $seeddata"_x.npy" $seeddata"_y.npy"
	fi

	# Update model counter
	COUNTER=$[$COUNTER +1]

	# Add this model to bag
	mv $seeddata"model" $bagfolder/$COUNTER

	# Keras specific change to make sure target model can be loaded in future
	# python fix.py $bagfolder/$COUNTER

	# Finetine proxy (make it adapt)
	python ../Code/train_model.py --mode finetune --nb_epochs 100 --learning_rate $lr --save_here $seedproxy --proxy_x $prefix"_xproxy.npy" --proxy_y $prefix"_yproxy.npy" --level blackbox --dataset $dataset

	# Keras specific change to make sure proxy model can be loaded in future
        # python fix.py $seedproxy

done < $order
