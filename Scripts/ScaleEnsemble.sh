#!/bin/bash

#python ../Code/cross_test.py --model_path $2 --proxy_data True --per_class_adv $3
python ../Code/cross_test.py --model_path $2 --proxy_data True --per_class_adv $3 --is_autoencoder 3

#Generate new data
cp PY.npy farziY.npy
cp PX.npy farziX0.npy
python ../Code/scale_images.py PX.npy 5 farziX5.npy
python ../Code/scale_images.py PX.npy 10 farziX10.npy
python ../Code/scale_images.py PX.npy -5 farziX-5.npy
python ../Code/scale_images.py PX.npy -10 farziX-10.npy


#Train proxy models
python ../Code/train_model.py --is_blackbox False --save_here $1cnn0r --batch_size 16 --learning_rate 0.005 --nb_epochs 50 --proxy_level 1 --proxy_x farziX0.npy --proxy_y farziY.npy
python ../Code/train_model.py --is_blackbox False --save_here $1cnn5r --batch_size 16 --learning_rate 0.005 --nb_epochs 50 --proxy_level 1 --proxy_x farziX5.npy --proxy_y farziY.npy
python ../Code/train_model.py --is_blackbox False --save_here $1cnn10r --batch_size 16 --learning_rate 0.005 --nb_epochs 50 --proxy_level 1 --proxy_x farziX10.npy --proxy_y farziY.npy
python ../Code/train_model.py --is_blackbox False --save_here $1cnn-5r --batch_size 16 --learning_rate 0.005 --nb_epochs 50 --proxy_level 1 --proxy_x farziX-5.npy --proxy_y farziY.npy
python ../Code/train_model.py --is_blackbox False --save_here $1cnn-10r --batch_size 16 --learning_rate 0.005 --nb_epochs 50 --proxy_level 1 --proxy_x farziX-10.npy --proxy_y farziY.npy


#Clean up
rm PX.npy
rm PY.npy
rm farziX0.npy
rm farziX5.npy
rm farziX10.npy
rm farziX-5.npy
rm farziX-10.npy
rm farziY.npy
