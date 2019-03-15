#!/bin/bash
export TF_CPP_MIN_LOG_LEVEL="2"
dataset=$1    # cifar10/mnist/svhn
seedmodel=$2  # path to starting model
bagfolder=$3  # new folder will be made to store bag of models here
order=$4      # file containing order of attacks
transfer=$5   # transfer of parameters (yes/no)
temp=$(date -d "today" +"%s")
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
python fix.py $seedmodel
cp $seedmodel $bagfolder/1
COUNTER=1 # Counting models
seeddata=$(date -d "today" +"%s") # Required if cumulative data is to be used
sleep 2                           # Make sure name does not clash with prefix inside loop
attackssofar=""                   # Keep track of attacks so far
while read attack
do
	echo "Running attack $COUNTER : $attack"
	prefix=$(date -d "today" +"%s") # Unique per dataset
	selectedmodel=$seedmodel
	if [ $transfer == "yes" ]; then
		selectedmodel=$bagfolder/$COUNTER
	fi
	cp $selectedmodel $seeddata"model"
	attackssofar="$attackssofar,$attack"
	if [ -n "$attackssofar" ]; then
		python ../Code/bagging.py --nb_epochs 150 --mode finetune --dataset $dataset --seed_model $seeddata"model" --model_dir $bagfolder --attack $attackssofar
	else
		python ../Code/bagging.py --nb_epochs 150 --mode finetune --dataset $dataset --seed_model $seeddata"model" --model_dir $bagfolder
	fi
	COUNTER=$[$COUNTER +1]
	mv $seeddata"model" $bagfolder/$COUNTER
done < $order