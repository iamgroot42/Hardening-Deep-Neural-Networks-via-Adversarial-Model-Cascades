#!/bin/bash

bash ScaleEnsemble.sh EnsembleModelsRotate/$1/10/ BM 10 >> LOG10
bash ScaleEnsemble.sh EnsembleModelsRotate/$1/25/ BM 25 >> LOG25
bash ScaleEnsemble.sh EnsembleModelsRotate/$1/75/ BM 75 >> LOG75
bash ScaleEnsemble.sh EnsembleModelsRotate/$1/100/ BM 100 >> LOG100
