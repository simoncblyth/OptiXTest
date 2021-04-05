#!/bin/bash 

source ../env.sh 

echo PREFIX $PREFIX

CUDA_PREFIX=/usr/local/cuda   # just use some CUDA headers, not using GPU 
GLM_PREFIX=$PREFIX/externals/glm/glm


#opts="-DDEBUG=1"
opts=""


name=GeoTest
srcs="GeoTest.cc ../Geo.cc ../Sys.cc ../Util.cc ../Grid.cc ../Foundry.cc ../Solid.cc ../Prim.cc ../Node.cc ../CU.cc ../Tran.cc"


gcc $srcs -std=c++11 \
      -I.. \
      -I${CUDA_PREFIX}/include  \
      -I${GLM_PREFIX} \
      -L${CUDA_PREFIX}/lib -lcudart \
      -lstdc++ $opts \
      -o /tmp/$name 

[ $? -ne 0 ] && echo compile error && exit 1


case $(uname) in
  Darwin) var=DYLD_LIBRARY_PATH ;;
  Linux)  var=LD_LIBRARY_PATH ;;
esac

mkdir -p /tmp/GeoTest_ 

cmd="$var=${CUDA_PREFIX}/lib /tmp/$name $*"
echo $cmd
eval $cmd
[ $? -ne 0 ] && echo run error && exit 2


exit 0 

