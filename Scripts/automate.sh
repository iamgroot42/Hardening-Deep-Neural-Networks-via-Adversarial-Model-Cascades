#!/bin/bash

ns=$1

#for i in $(cat epsilon_values)
#do
#	bash pipeline.sh $i $ns
#done

#rm *.npy

#for i in $(cat epsilon_values)
#do
#	bash pipeline_hp.sh $i $ns
#done

#rm *.npy

for i in $(cat epsilon_values)
do
	bash pipeline_ae.sh $i $ns
done

rm *.npy
