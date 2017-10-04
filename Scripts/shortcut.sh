#!/bin/bash

folder=$1
dataset=$2
prefix="${folder::-1}"

bash testFGSMbag.sh $dataset $folder 10 Proxy/PM10 >> "10"$prefix
bash testFGSMbag.sh $dataset $folder 25 Proxy/PM25 >> "25"$prefix
bash testFGSMbag.sh $dataset $folder 40 Proxy/PM40 >> "40"$prefix
bash testFGSMbag.sh $dataset $folder 60 Proxy/PM60 >> "60"$prefix
