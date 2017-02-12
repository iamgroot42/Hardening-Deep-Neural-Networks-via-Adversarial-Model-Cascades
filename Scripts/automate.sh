#!/bin/bash

ns=$2

for i in $(cat epsilon_values)
do
	bash pipeline.sh $i $ns
	#bash pipeline_hp.sh $i $ns
	#bash pipeline_ae.sh $i $ns
	#bash pipeline_hyb.sh $i $ns
done
