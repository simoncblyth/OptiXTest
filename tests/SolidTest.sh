#!/bin/bash -l 

name=SolidTest 
srcs="$name.cc ../Solid.cc"


gcc -g \
   $srcs \
   -I.. \
   -lstdc++ -std=c++11 \
   -I/usr/local/cuda/include \
   -o /tmp/$name 

[ $? -ne 0 ] && echo compile error && exit 1


/tmp/$name $*
[ $? -ne 0 ] && echo run error && exit 2


exit 0
