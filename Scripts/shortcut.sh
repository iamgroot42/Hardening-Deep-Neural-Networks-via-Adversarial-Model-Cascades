#!/bin/bash

folder=$1
prefix="${folder::-1}"

bash testFGSMbag.sh mnist $folder 10 >> "10"$prefix
bash testFGSMbag.sh mnist $folder 25 >> "25"$prefix
bash testFGSMbag.sh mnist $folder 75 >> "75"$prefix
bash testFGSMbag.sh mnist $folder 100 >> "100"$prefix
