#!/bin/sh

file="pid.txt"
read LINE < $file
echo "pid : $LINE" 
kill -2 $LINE
