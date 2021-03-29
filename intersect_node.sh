#!/bin/bash 

CUDA_PREFIX=/usr/local/cuda   # just use some CUDA headers, not using GPU 


#opts="-DDEBUG=1"
opts=""

name=intersect_node 
gcc $name.cc Solid.cc Scan.cc  -std=c++11 -lstdc++ $opts -I. -I$HOME/np -I${CUDA_PREFIX}/include -o /tmp/$name 
[ $? -ne 0 ] && echo compile error && exit 1


scans="axis rectangle circle"
for scan in $scans ; do 
    tmpdir=/tmp/intersect_node_tests/${scan}_scan
    mkdir -p $tmpdir 
done 

/tmp/$name $*
[ $? -ne 0 ] && echo run error && exit 2

for scan in $scans ; do 
    tmpdir=/tmp/intersect_node_tests/${scan}_scan
    ls -l $tmpdir
done 


exit 0 
