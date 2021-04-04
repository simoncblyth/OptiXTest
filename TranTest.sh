#!/bin/bash -l

source ./env.sh 

CUDA_PREFIX=/usr/local/cuda   # just use some CUDA headers, not using GPU 

#opts="-DDEBUG=1"
opts=""

name=TranTest
gcc -g $name.cc Tran.cc \
       -std=c++11 \
       -I. \
       -I${CUDA_PREFIX}/include \
       -I$PREFIX/externals/glm/glm \
       -lstdc++ \
       $opts \
       -o /tmp/$name 

[ $? -ne 0 ] && echo compile error && exit 1

cmd="/tmp/$name $*"
echo $cmd
eval $cmd
[ $? -ne 0 ] && echo run error && exit 2

exit 0 

