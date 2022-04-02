#!/bin/bash

for job in $(seq 3633458 3633487):
do
qdel $job
done 
