#!/usr/bin/env nextflow

process PrepData {
    // beforeScript source activate tensorEnv
    publishDir params.saveDir, mode: params.publish_dir_mode,
                enabled: params.save
    
    label  "with_cpus"
    tag    "Data Cleaning"
    
    input:
        path prot
        path ncbiVirus
        path eID
        path virusDB
        path fasta

    output:
        path "*"
    
    script:
        """
        data_cleaning.py --uniprot ${prot} \
                            --ncbivirusdb ${ncbiVirus} \
                            --liverpooluni ${eID} --virusdb ${virusDB} --fasta ${fasta}
        """
}