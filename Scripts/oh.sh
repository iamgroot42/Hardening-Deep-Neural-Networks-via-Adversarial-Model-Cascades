#!/bin/bash


python ../Code/train_model.py --is_blackbox False --save_here PM --batch_size 16 --learning_rate 0.01 --nb_epochs 300
mv PM PM.01

python ../Code/train_model.py --is_blackbox False --save_here PM --batch_size 16 --learning_rate 0.001 --nb_epochs 300
mv PM PM.001

