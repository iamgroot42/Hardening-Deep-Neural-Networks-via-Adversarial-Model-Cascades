#!/bin/bash

for i in $(cat epsilon_values)
do
	bash pipeline.sh $i
	#bash pipeline_hp.sh $i
	#bash pipeline_ae.sh $i
	#bash pipeline_hyb.sh $i
done
