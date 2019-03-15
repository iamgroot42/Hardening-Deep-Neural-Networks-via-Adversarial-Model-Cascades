#!/bin/bash
export TF_CPP_MIN_LOG_LEVEL="2"
dataset=$1
fsqueeze=$2
sap=$3
modelpath=$4
attack=$5
proxymodelpath=$6
if [ $dataset == "mnist" ]; then
                :
elif [ $dataset == "svhn" ]; then
                :
elif [ $dataset == "cifar10" ]; then
              	:
else
        echo "Invalid dataset! Exiting"
        exit
fi
temp=$(date -d "today" +"%s")
if [ $proxymodelpath == "no" ]; then
	python ../Code/attack.py --dataset $dataset --model $modelpath --attack_name $attack --save_here $temp --mode attack
else
	python ../Code/attack_proxy.py --dataset $dataset --model $proxymodelpath --attack_name $attack --save_here $temp --mode attack
fi
if [ $fsqueeze == "yes" ]; then
		python ../Code/feature_squeeze.py --dump_data $temp"_x.npy" --load_data $temp"_x.npy"
fi
if [ $sap == "yes" ]; then
	python ../Code/sap_predict.py -dx $temp"_x.npy" -dy $temp"_y.npy" -m $modelpath -f 1.0 -s 100 -d $dataset
else
	python ../Code/test_accuracy.py --test_prefix $temp --dataset $dataset  --model_path $modelpath
fi
rm $temp"_x.npy" $temp"_y.npy"