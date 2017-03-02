#!/bin/bash

bash automate.sh 100
mv ../Data/Outputs ../Data/CNNSep100
bash automate.sh 500
mv ../Data/Outputs ../Data/CNNSep500
bash automate.sh 1000
mv ../Data/Outputs ../Data/CNNSep1000
bash automate.sh 2500
mv ../Data/Outputs ../Data/CNNSep2500
