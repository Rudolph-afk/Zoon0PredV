#!/usr/bin/env nextflow

process ChaosGameRepresentation {
    // publishDir "${directory}/human-true", mode: params.publish_dir_mode, 
    //             enabled: params.save, pattern: "human_true*"
    // publishDir "${directory}/human-false", mode: params.publish_dir_mode, 
    //             enabled: params.save, pattern: "human_false*"


    input:
        // tuple val(parentDir), path(directory), path(fastaFile)
        path directory

    output:
        path "${directory}"

    script:
    if (params.trainOnly) {
        """
        cd ${directory}/train/
        chaos_game_representation_of_protein_sequences.R --fasta \*.fasta # --directory \${directory} assumes only a single fasta file in directory and the file extension is .fasta
        """
    } else if (params.testOnly) {
        """
        cd ${directory}/test/
        chaos_game_representation_of_protein_sequences.R --fasta \*.fasta # --directory \${directory} assumes only a single fasta file in directory and the file extension is .fasta
        """
    } else {
        """
        cd ${directory}/train/
        chaos_game_representation_of_protein_sequences.R --fasta \*.fasta # --directory \${directory} assumes only a single fasta file in directory and the file extension is .fasta
        cd ../../${directory}/test/
        chaos_game_representation_of_protein_sequences.R --fasta \*.fasta # --directory \${directory} assumes only a single fasta file in directory and the file extension is .fasta
        """
    }
}

// process MoveCGRImages {

//     label "with_cpus"
//     cache "lenient"

//     input:
//         path parentDir
    
//     output:
//         val "${parentDir}"

//     script:
//         """
//         find $parentDir/test -name 'human_true*' -type f -exec mv {} $parentDir/test/human-true \\;
//         find $parentDir/train -name 'human_true*' -type f -exec mv {} $parentDir/train/human-true \\;
//         find $parentDir/test -name 'human_false*' -type f -exec mv {} $parentDir/test/human-false \\;
//         find $parentDir/train -name 'human_false*' -type f -exec mv {} $parentDir/train/human-false \\;
//         """
// }