#!/bin/bash

bash ensemble_scale.sh EnsembleModelsRotate/$1 MMData/ 0.000 BM > 0.000
bash ensemble_scale.sh EnsembleModelsRotate/$1 MMData/ 0.005 BM > 0.005
bash ensemble_scale.sh EnsembleModelsRotate/$1 MMData/ 0.010 BM > 0.010
bash ensemble_scale.sh EnsembleModelsRotate/$1 MMData/ 0.015 BM > 0.015
bash ensemble_scale.sh EnsembleModelsRotate/$1 MMData/ 0.020 BM > 0.020
bash ensemble_scale.sh EnsembleModelsRotate/$1 MMData/ 0.025 BM > 0.025
bash ensemble_scale.sh EnsembleModelsRotate/$1 MMData/ 0.040 BM > 0.030
bash ensemble_scale.sh EnsembleModelsRotate/$1 MMData/ 0.040 BM > 0.040
bash ensemble_scale.sh EnsembleModelsRotate/$1 MMData/ 0.050 BM > 0.050
bash ensemble_scale.sh EnsembleModelsRotate/$1 MMData/ 0.100 BM > 0.100
