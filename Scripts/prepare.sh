#!/bin/bash

#mkdir -p ../Code/SVHN
cd ../Code/SVHN

# Processed SVHN
wget https://www.dropbox.com/s/3a5xs5944wcvjyg/SVHNx_tr.npy?dl=1 -O SVHNx_tr.npy
wget https://www.dropbox.com/s/btq7yfxsy7ybfyh/SVHNy_tr.npy?dl=1 -O SVHNy_tr.npy

wget https://www.dropbox.com/s/576h9kg3ne0zoqq/SVHNx_te.npy?dl=1 -O SVHNx_te.npy
wget https://www.dropbox.com/s/sw9nn9rhkzai7i2/SVHNy_te.npy?dl=1 -O SVHNy_te.npy

cd ../../Scripts
echo $(pwd)

# Unlaballed data for proxy
wget https://www.dropbox.com/s/2n0sj9j3okaocvl/UnlabelledData.tar.gz?dl=1 -O UnlabelledData.tar.gz
tar -zxvf UnlabelledData.tar.gz
rm UnlabelledData.tar.gz
mv ProxyUnlabelledData UnlabelledData

# MNIST Models
wget https://www.dropbox.com/s/8q1u7wdcqybzcde/MNISTModels.tar.gz?dl=1 -O MNISTModels.tar.gz
tar -zxvf MNISTModels.tar.gz
rm MNISTModels.tar.gz

# SVHN Models
wget https://www.dropbox.com/s/eue9d8y1da893mg/SVHNModels.tar.gz?dl=1 -O SVHNModels.tar.gz
tar -zxvf SVHNModels.tar.gz
rm SVHNModels.tar.gz
