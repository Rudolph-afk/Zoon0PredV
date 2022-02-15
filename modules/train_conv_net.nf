#!/usr/bin/env nextflow

process ModelTraining {
    
    publishDir "${params.saveDir}/${directory}", mode: params.publish_dir_mode,
                enabled: params.save

    tag        "${directory}"
    maxForks   1
    
    input:
        path   directory

    output:
        path   "model"
        path   "*.csv"
    
    script:
        """
        train_zoonosis_model.py -d ${directory} 
        """
}