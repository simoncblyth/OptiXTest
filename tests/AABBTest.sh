#!/bin/bash -l

CUDA_PREFIX=/usr/local/cuda   # just use some CUDA headers, not using GPU 

name=AABBTest
srcs=$name.cc 

gcc -g \
    $srcs \
    -I.. \
    -I${CUDA_PREFIX}/include \
    -std=c++11  -lstdc++ \
    -L${CUDA_PREFIX}/lib -lcudart  \
    -o /tmp/$name 

[ $? -ne 0 ] && echo compile error && exit 1


cmd="/tmp/$name $*"
echo $cmd
eval $cmd
[ $? -ne 0 ] && echo run error && exit 2

exit 0 

