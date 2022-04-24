#!/bin/bash

echo Running experiment: $1
echo Number of views: $2
echo Number of validation views: $3
echo Load path ${@:4}

python3 train.py -n $2 -e $1 -v $3 -p $4 > run.log 2>&1 &
echo Detached

