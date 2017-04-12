#!/bin/bash

rm PM
bash automate.sh 10
mv ../Data/VanillaCNN ../Data/VanillaCNN10_
rm PM
bash automate.sh 25
mv ../Data/VanillaCNN ../Data/VanillaCNN25_
rm PM
bash automate.sh 75
mv ../Data/VanillaCNN ../Data/VanillaCNN75_
rm PM
bash automate.sh 100
mv ../Data/VanillaCNN ../Data/VanillaCNN100_
rm PM
