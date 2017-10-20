#!/bin/bash

mkdir -p ../Code/SVHN
cd ../Code/SVHN

#wget https://www.dropbox.com/s/3a5xs5944wcvjyg/SVHNx_tr.npy?dl=1 -O SVHNx_tr.npy
#wget https://www.dropbox.com/s/btq7yfxsy7ybfyh/SVHNy_tr.npy?dl=1 -O SVHNy_tr.npy

#wget https://www.dropbox.com/s/576h9kg3ne0zoqq/SVHNx_te.npy?dl=1 -O SVHNx_te.npy
#wget https://www.dropbox.com/s/sw9nn9rhkzai7i2/SVHNy_te.npy?dl=1 -O SVHNy_te.npy

cd ../../Scripts
echo $(pwd)

wget https://www.dropbox.com/s/f9dbh0i9s36gvtn/ProxyUnlabelledData.tar.gz?dl=1 -O ProxyUnlabelledData.tar.gz
tar -zxvf ProxyUnlabelledData.tar.gz
rm ProxyUnlabelledData.tar.gz
