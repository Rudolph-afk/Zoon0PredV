#!/usr/bin/env nextflow
nextflow.enable.dsl=2

process PrepData {
    // beforeScript source activate tensorEnv
    label "with_cpus"

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
        data_cleaning.py --uniprot ${prot} --ncbivirusdb ${ncbiVirus} --liverpooluni ${eID} --virusdb ${virusDB} --fasta ${fasta}
        """
}