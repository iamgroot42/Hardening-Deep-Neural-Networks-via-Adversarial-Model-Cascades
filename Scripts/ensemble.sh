#!/bin/bash

# Generate union, intersection models
python ../Code/mixture_attack.py --models_directory $1 --fgsm_eps $3 --models_data_directory $2
# Move labels outside directory
mv $2labels.npy labels.npy
mv $2labels_union.npy labels_union.npy
mv $2labels_intersection.npy labels_intersection.npy
# Run for all the data, report misclassification accuracies

for model in $(ls $2)
do
	if [ $model != "union.npy" ] && [ $model != "intersection.npy" ]
	then
		echo "For" $model
		python ../Code/cross_test.py --model_path $4 --adversary_path_x $2$model --adversary_path_y labels.npy
	fi
done

echo "For union.npy"
python ../Code/cross_test.py --model_path $4 --adversary_path_x $2"union.npy" --adversary_path_y labels_union.npy

echo "For intersection.npy"
python ../Code/cross_test.py --model_path $4 --adversary_path_x $2"intersection.npy" --adversary_path_y labels_intersection.npy
