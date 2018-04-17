#!/bin/sh

file="pid.txt"
read LINE < $file
echo "pid : $LINE" 
sudo kill -2 $LINE
