#!/bin/bash -l 

name=$(basename $PWD)
cmd="rsync -rtz --progress --exclude='.git/' $PWD/ P:$name/"
echo $cmd
eval $cmd
