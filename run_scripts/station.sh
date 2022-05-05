#!/bin/bash

##? Ilifu: Run in login shell and not in interactive shell else you'll be kicked out

module load graphviz # required for DAG

cd ..

nextflow run main.nf \
            -profile station \
            -resume \
            -params-file params/params.yml # -N 4056876@myuwc.ac.za
