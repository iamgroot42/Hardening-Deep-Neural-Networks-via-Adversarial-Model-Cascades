#!/bin/bash

# bash pipeline.sh 0.105
# bash pipeline.sh 0.115
# bash pipeline.sh 0.125
# bash pipeline.sh 0.135
# bash pipeline.sh 0.145

bash pipeline_hp.sh 0.10 5
bash pipeline_hp.sh 0.10 10
bash pipeline_hp.sh 0.10 20
bash pipeline_hp.sh 0.10 40
bash pipeline_hp.sh 0.10 80
bash pipeline_hp.sh 0.10 160
