#!/bin/bash -l 

name=InstanceIdTest 
gcc $name.cc \
    -I.. \
    -std=c++11 \
    -lstdc++ -o /tmp/$name && /tmp/$name

