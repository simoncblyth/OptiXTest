#!/bin/bash 

source ./env.sh 

CUDA_PREFIX=/usr/local/cuda   # just use some CUDA headers, not using GPU 

#opts="-DDEBUG=1"
opts=""

name=ScanTest 
gcc $name.cc Foundry.cc Solid.cc Prim.cc Node.cc Scan.cc CU.cc Tran.cc \
          -std=c++11 \
          $opts \
          -I. \
          -I${CUDA_PREFIX}/include \
          -I$PREFIX/externals/glm/glm \
          -L${CUDA_PREFIX}/lib -lcudart -lstdc++ \
          -o /tmp/$name 

[ $? -ne 0 ] && echo compile error && exit 1
#exit 0

base=/tmp/ScanTest_scans

scans="axis rectangle circle"
for scan in $scans ; do 
    tmpdir=$base/${scan}_scan
    mkdir -p $tmpdir 
done 


case $(uname) in
  Darwin) var=DYLD_LIBRARY_PATH ;;
  Linux)  var=LD_LIBRARY_PATH ;;
esac
cmd="$var=${CUDA_PREFIX}/lib /tmp/$name $*"
echo $cmd
eval $cmd
[ $? -ne 0 ] && echo run error && exit 2


scan-all()
{
    echo $FUNCNAME $*
    local scan
    for scan in $* ; do 
       tmpdir=$base/${scan}_scan
       echo $tmpdir
       ls -l $tmpdir
    done 
}
scan-recent(){
   echo $FUNCNAME 
   find $base -newer ScanTest.cc -exec ls -l {} \; 
}

#scan-all $scans
scan-recent 



exit 0 
