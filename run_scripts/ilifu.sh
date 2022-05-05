#!/bin/bash

##? Ilifu: Run in login shell and not in interactive shell else you'll be kicked out
##? Before pipeline completion

module load graphviz # required for DAG

cd ..

echo $1
echo $2

nextflow run main.nf \
            -profile ilifu \
            -params-file $1 $2 # -N 4056876@myuwc.ac.za

