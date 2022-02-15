#!/usr/bin/env nextflow
<<<<<<< HEAD

process ChaosGameRepresentation {
    // publishDir "${directory}/human-true", mode: params.publish_dir_mode, 
    //             enabled: params.save, pattern: "human_true*"
    // publishDir "${directory}/human-false", mode: params.publish_dir_mode, 
    //             enabled: params.save, pattern: "human_false*"
=======
nextflow.enable.dsl=2 // Required as some methods/functions may not work in main script e.g. flatten()

process ChaosGameRepresentation {
    publishDir "${params.saveDir}/${parentDir}/${directory}/human-true/",
                mode: params.publish_dir_mode, 
                enabled: params.save, pattern: "human_true*"
                
    publishDir "${params.saveDir}/${parentDir}/${directory}/human-false/",
                mode: params.publish_dir_mode, 
                enabled: params.save, pattern: "human_false*"
>>>>>>> 6377fef3073d3822750dc9f1941ae577b02fdca9

    maxForks 3
    tag "${parentDir}(${directory})"

    input:
        // tuple val(parentDir), path(directory), path(fastaFile)
        path directory

    output:
<<<<<<< HEAD
        path "${directory}"
=======
        path "*.jpeg" // Collects all jpeg files outputted
        val "${parentDir}" // Relevant as a reference to the next process in the main script (training/testing)
>>>>>>> 6377fef3073d3822750dc9f1941ae577b02fdca9

    script:
    if (params.trainOnly) {
        """
<<<<<<< HEAD
        cd ${directory}/train/
        chaos_game_representation_of_protein_sequences.R --fasta \*.fasta # --directory \${directory} assumes only a single fasta file in directory and the file extension is .fasta
=======
        chaos_game_representation_of_protein_sequences.R --fasta ${fastaFile}
>>>>>>> 6377fef3073d3822750dc9f1941ae577b02fdca9
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

<<<<<<< HEAD
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
=======
// May add more features
>>>>>>> 6377fef3073d3822750dc9f1941ae577b02fdca9
