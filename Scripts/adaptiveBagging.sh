#!/bin/bash
export TF_CPP_MIN_LOG_LEVEL="2"
dataset=$1    # cifar10/mnist/svhn
seedmodel=$2  # path to starting model
bagfolder=$3  # new folder will be made to store bag of models here
order=$4      # file containing order of attacks
transfer=$5   # transfer of parameters (yes/no)
seedproxy=$6  # path to starting proxy model
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
cp $seedmodel $bagfolder/1
echo "Seed proxy model will be stored by the name $seedproxy.hdf5"
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
	python ../Code/train_model_proxy.py -e 100 -l 0.1 -s $seedproxy -m finetune -d $dataset -k True -t $bagfolder/$COUNTER -b 64 -a $attackssofar
	python ../Code/attack_proxy.py --dataset $dataset --batch_size 128 --model $seedproxy --attack_name $attackssofar --save_here $prefix --multiattacks --mode harden
	python ../Code/bagging.py --nb_epochs 150 --mode finetune --dataset $dataset --seed_model $seeddata"model" --model_dir $bagfolder --data_x $prefix"_x.npy" --data_y $prefix"_y.npy" --attack $attackssofar
	COUNTER=$[$COUNTER +1]
	mv $seeddata"model" $bagfolder/$COUNTER
done < $order
echo "Saving adaptive proxy as $seedproxy.hdf5"
mv $seedproxy "$seedproxy.hdf5"