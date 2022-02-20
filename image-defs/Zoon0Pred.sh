#!/bin/sh

nextflow run main.nf -resume -dsl2 \
    -with-report \
    -N "4056876@myuwc.ac.za"
