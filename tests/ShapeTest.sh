#!/bin/bash -l 

source ../env.sh 
name=ShapeTest 
mkdir -p /tmp/ShapeTestWrite/{0,1} 

srcs="$name.cc ../Node.cc ../Shape.cc ../Sys.cc"

gcc -g \
      $srcs \
      -I.. \
      -lstdc++ -std=c++11 \
      -I$PREFIX/externals/glm/glm \
      -I/usr/local/cuda/include \
      -o /tmp/$name 
[ $? -ne 0 ] && echo compile error && exit 1 

/tmp/$name
[ $? -ne 0 ] && echo run error && exit 2

exit 0  
