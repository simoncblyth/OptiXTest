#!/bin/bash 

CUDA_PREFIX=/usr/local/cuda   # just use some CUDA headers, not using GPU 

#opts="-DDEBUG=1"
opts=""

name=FoundryTest
gcc $name.cc Foundry.cc  -std=c++11 -lstdc++ $opts -I. -I$HOME/np -I${CUDA_PREFIX}/include -o /tmp/$name 
[ $? -ne 0 ] && echo compile error && exit 1


/tmp/$name $*
[ $? -ne 0 ] && echo run error && exit 2


exit 0 
