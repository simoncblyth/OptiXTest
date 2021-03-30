#!/bin/bash 

CUDA_PREFIX=/usr/local/cuda   # just use some CUDA headers, not using GPU 


#opts="-DDEBUG=1"
opts=""

name=ScanTest 
gcc $name.cc Foundry.cc Solid.cc Prim.cc Node.cc Scan.cc  -std=c++11 -lstdc++ $opts -I. -I${CUDA_PREFIX}/include -o /tmp/$name 
[ $? -ne 0 ] && echo compile error && exit 1

base=/tmp/ScanTest_scans

scans="axis rectangle circle"
for scan in $scans ; do 
    tmpdir=$base/${scan}_scan
    mkdir -p $tmpdir 
done 

/tmp/$name $*
[ $? -ne 0 ] && echo run error && exit 2

for scan in $scans ; do 
    tmpdir=$base/${scan}_scan
    echo $tmpdir
    ls -l $tmpdir
done 


exit 0 
