#!/bin/bash

python ../Code/cross_test.py --model_path $2 --proxy_data True --per_class_adv $3

#Generate new data
cp PY.npy farziY.npy
cp PX.npy farziX1.npy
python ../Code/scale_images.py PX.npy 2 farziX2.npy
python ../Code/scale_images.py PX.npy 4 farziX4.npy

#Train proxy models
python ../Code/train_model.py --is_blackbox False --save_here $1cnn1x --batch_size 16 --learning_rate 0.00005 --nb_epochs 50 --proxy_level 1 --proxy_x farziX1.npy --proxy_y farziY.npy
python ../Code/train_model.py --is_blackbox False --save_here $1cnn2x --batch_size 16 --learning_rate 0.00005 --nb_epochs 50 --proxy_level 2 --proxy_x farziX2.npy --proxy_y farziY.npy
python ../Code/train_model.py --is_blackbox False --save_here $1cnn4x --batch_size 16 --learning_rate 0.00005 --nb_epochs 50 --proxy_level 4 --proxy_x farziX4.npy --proxy_y farziY.npy

#Clean up
rm PX.npy
rm PY.npy
rm farziX1.npy
rm farziX2.npy
rm farziX4.npy
rm farziY.npy

