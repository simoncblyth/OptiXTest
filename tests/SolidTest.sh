#!/bin/bash -l 

name=SolidTest 
srcs="$name.cc ../Solid.cc"

gcc $srcs -I.. -std=c++11 -lstdc++ -o /tmp/$name && /tmp/$name 

