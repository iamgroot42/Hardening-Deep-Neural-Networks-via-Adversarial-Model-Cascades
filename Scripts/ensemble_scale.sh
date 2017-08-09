#!/bin/bash

# Generate union, intersection models
python ../Code/mixture_attack.py --models_directory $1 --fgsm_eps $3 --models_data_directory $2
# Move labels outside directory
mv $2labels.npy labels.npy
mv $2labels_union.npy labels_union.npy
mv $2labels_intersection.npy labels_intersection.npy
# Run for all the data, report misclassification accuracies

echo "For cnn0r.npy"
python ../Code/cross_test.py --model_path $4 --adversary_path_x $2"cnn0r.npy" --adversary_path_y labels.npy #--is_autoencoder 3

echo "For cnn5r.npy"
python ../Code/cross_test.py --model_path $4 --adversary_path_x $2"cnn5r.npy" --adversary_path_y labels.npy #--is_autoencoder 3

echo "For cnn10r.npy"
python ../Code/cross_test.py --model_path $4 --adversary_path_x $2"cnn10r.npy" --adversary_path_y labels.npy #--is_autoencoder 3

echo "For cnn-5r.npy"
python ../Code/cross_test.py --model_path $4 --adversary_path_x $2"cnn-5r.npy" --adversary_path_y labels.npy #--is_autoencoder 3

echo "For cnn-10r.npy"
python ../Code/cross_test.py --model_path $4 --adversary_path_x $2"cnn-10r.npy" --adversary_path_y labels.npy #--is_autoencoder 3

echo "For union.npy"
python ../Code/cross_test.py --model_path $4 --adversary_path_x $2"union.npy" --adversary_path_y labels_union.npy #--is_autoencoder 3

echo "For intersection.npy"
python ../Code/cross_test.py --model_path $4 --adversary_path_x $2"intersection.npy" --adversary_path_y labels_intersection.npy #--is_autoencoder 3

rm $2"cnn0r.npy"
rm $2"cnn5r.npy"
rm $2"cnn10r.npy"
rm $2"cnn-5r.npy"
rm $2"cnn-10r.npy"
rm labels.npy
rm labels_union.npy
rm labels_intersection.npy
