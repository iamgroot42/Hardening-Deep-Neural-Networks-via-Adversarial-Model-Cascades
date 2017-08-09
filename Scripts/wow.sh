#!/bin/bash

mkdir pot10 pot25 pot75 pot100

bash aalas.sh $1/10/
mv 0.* pot10/
bash aalas.sh $1/25/
mv 0.* pot25/
bash aalas.sh $1/75/
mv 0.* pot75/
bash aalas.sh $1/100/
mv 0.* pot100/
