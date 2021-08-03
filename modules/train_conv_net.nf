#!/usr/bin/env nextflow
nextflow.enable.dsl=2

process ModelTraining {
    if (params.save) {
        publishDir "${params.saveDir}", mode: "${params.publish_dir_mode}"
    }

    beforeScript "source activate tensorEnv"
    
    label "with_gpus" // (params.GPU == "ON" ? "": "with_cpus")
    
    input:
        path directory

    output:
        path "${directory}"
    
    script:
        """
        train_zoonosis_model.py -d ${directory} 
        """
}