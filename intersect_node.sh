#!/bin/bash 

CUDA_PREFIX=/usr/local/cuda   # just use some CUDA headers, not using GPU 

name=intersect_node 
gcc $name.cc -std=c++11 -lstdc++ -I. -I${CUDA_PREFIX}/include -o /tmp/$name 
[ $? -ne 0 ] && echo compile error && exit 1

/tmp/$name
[ $? -ne 0 ] && echo run error && exit 2

exit 0 
