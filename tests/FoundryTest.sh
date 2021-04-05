#!/bin/bash -l

source ../env.sh 

CUDA_PREFIX=/usr/local/cuda   # just use some CUDA headers, not using GPU 

#opts="-DDEBUG=1"
opts=""

name=FoundryTest
srcs="$name.cc ../Foundry.cc ../Solid.cc ../Prim.cc ../PrimSpec.cc ../Node.cc ../CU.cc ../Tran.cc ../Util.cc"
echo compiling $srcs
gcc -g \
       $srcs \
       -std=c++11 \
       -I.. \
       -I${CUDA_PREFIX}/include \
       -I$PREFIX/externals/glm/glm \
       -L${CUDA_PREFIX}/lib -lcudart \
       -lstdc++ $opts \
       -o /tmp/$name 

[ $? -ne 0 ] && echo compile error && exit 1
echo compile done 

case $(uname) in
  Darwin) var=DYLD_LIBRARY_PATH debugger=lldb_  ;;
  Linux)  var=LD_LIBRARY_PATH   debugger=gdb    ;;
esac

echo var $var debugger $debugger

mkdir -p /tmp/FoundryTest_

cmd="$var=${CUDA_PREFIX}/lib  /tmp/$name $*"
echo $cmd
eval $cmd
[ $? -ne 0 ] && echo run error && exit 2


exit 0 

