#!/bin/bash -l 

name=GridTest 
srcs="$name.cc ../Grid.cc ../Util.cc"

mkdir -p /tmp/GridTestWrite/{0,1} 

glm- 

gcc -g  \
    $srcs \
    -I.. \
    -I$(glm-prefix) \
    -lstdc++ -std=c++11 \
    -o /tmp/$name 
[ $? -ne 0 ] && echo compile error && exit 1 


/tmp/$name
[ $? -ne 0 ] && echo run error && exit 2

exit 0 

