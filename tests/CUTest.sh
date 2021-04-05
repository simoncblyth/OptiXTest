#!/bin/bash 

CUDA_PREFIX=/usr/local/cuda  

name=CUTest

srcs="$name.cc ../CU.cc"
gcc $srcs \
    -I.. \
    -I${CUDA_PREFIX}/include \
    -std=c++11 \
    -L${CUDA_PREFIX}/lib -lcudart \
    -lstdc++ \
    -o /tmp/$name 
[ $? -ne 0 ] && echo compile error && exit 1

case $(uname) in
  Darwin) var=DYLD_LIBRARY_PATH ;;
  Linux)  var=LD_LIBRARY_PATH ;;
esac

cmd="$var=${CUDA_PREFIX}/lib /tmp/$name $*"
echo $cmd
eval $cmd
[ $? -ne 0 ] && echo run error && exit 2

exit 0 


