#!/usr/bin/env nextflow
nextflow.enable.dsl=2 // Required as some methods/functions may not work in main script e.g. flatten()

process ChaosGameRepresentation {
    publishDir "${params.saveDir}/${parentDir}/${directory}/human-true/",
                mode: params.publish_dir_mode, 
                enabled: params.save, pattern: "human_true*"
                
    publishDir "${params.saveDir}/${parentDir}/${directory}/human-false/",
                mode: params.publish_dir_mode, 
                enabled: params.save, pattern: "human_false*"

    maxForks 3
    tag "${parentDir}(${directory})"

    input:
        tuple val(parentDir), path(directory), path(fastaFile)

    output:
        path "*.jpeg" // Collects all jpeg files outputted
        val "${parentDir}" // Relevant as a reference to the next process in the main script (training/testing)

    script:
        """
        chaos_game_representation_of_protein_sequences.R --fasta ${fastaFile}
        """
}

// May add more features