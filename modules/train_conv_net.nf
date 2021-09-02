#!/usr/bin/env nextflow
nextflow.enable.dsl=2

process ModelTraining {
    publishDir "${directory}", mode: params.publish_dir_mode,
                enabled: params.save

    // beforeScript "source activate tensorEnv"
    maxForks 1
    
    input:
        path directory

    output:
        path "${directory}"
    
    script:
        """
        train_zoonosis_model.py -d ${directory} 
        """
}