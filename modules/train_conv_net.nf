#!/usr/bin/env nextflow

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