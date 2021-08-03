#!/usr/bin/env nextflow
nextflow.enable.dsl=2

process ChaosGameRepresentation {

    label "with_cpus"
    cache "lenient"

    input:
        tuple val(parentDir), path(directory), path(FastaFile)

    output:
        val "${parentDir}"

    script:
        """
        chaos_game_representation_of_protein_sequences.R --fasta $FastaFile --directory $directory
        find $directory -name 'human_true*' -type f -exec mv {} $directory/human-true \\;
        find $directory -name 'human_false*' -type f -exec mv {} $directory/human-false \\;
        """
}

process MoveCGRImages {

    label "with_cpus"
    cache "lenient"

    input:
        path parentDir
    
    output:
        val "${parentDir}"

    script:
        """
        find $parentDir/test -name 'human_true*' -type f -exec mv {} $parentDir/test/human-true \\;
        find $parentDir/train -name 'human_true*' -type f -exec mv {} $parentDir/train/human-true \\;
        find $parentDir/test -name 'human_false*' -type f -exec mv {} $parentDir/test/human-false \\;
        find $parentDir/train -name 'human_false*' -type f -exec mv {} $parentDir/train/human-false \\;
        """
}