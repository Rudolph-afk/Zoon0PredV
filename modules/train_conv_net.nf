#!/usr/bin/env nextflow

process ModelTraining {
    
    publishDir "${params.saveDir}/${directory}", mode: params.publish_dir_mode,
                enabled: params.save

    tag        "${directory}"
    label       "with_gpus"
    
    input:
        tuple  val(directory), path(train)

    output:
        tuple  val(directories), path("model")
        path   "*.csv"
    
    script:
        """
        tar -xzvf ${train}
        train_zoonosis_model.py -t .
        """
}