#!/bin/bash

rm PM
bash automate.sh 25
mv ../Data/CNNSVM ../Data/CNNSVM25
rm PM
bash automate.sh 75
mv ../Data/CNNSVM ../Data/CNNSVM75
rm PM
bash automate.sh 100
mv ../Data/CNNSVM ../Data/CNNSVM100
rm PM
