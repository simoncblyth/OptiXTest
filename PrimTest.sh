#!/bin/bash -l

CUDA_PREFIX=/usr/local/cuda   # just use some CUDA headers, not using GPU 

name=PrimTest
gcc -g $name.cc Prim.cc PrimSpec.cc CU.cc -std=c++11  -lstdc++ -L${CUDA_PREFIX}/lib -lcudart  -I.  -I${CUDA_PREFIX}/include -o /tmp/$name 
[ $? -ne 0 ] && echo compile error && exit 1


case $(uname) in
  Darwin) var=DYLD_LIBRARY_PATH debugger=lldb_  ;;
  Linux)  var=LD_LIBRARY_PATH   debugger=gdb    ;;
esac

cmd="$var=${CUDA_PREFIX}/lib  /tmp/$name $*"
echo $cmd
eval $cmd
[ $? -ne 0 ] && echo run error && exit 2


exit 0 

