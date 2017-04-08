#!/bin/bash

python ../Code/train_model.py --is_blackbox True --save_here BM --specialCNN sota --batch_size 16 --learning_rate 0.0001 --nb_epochs 400 >> 1
mv BM BM1

python ../Code/train_model.py --is_blackbox True --save_here BM --specialCNN sota --batch_size  16 --learning_rate 0.0005 --nb_epochs 400 >> 5
mv BM BM5
