nextflow.enable.dsl=2 // Required as some methods/functions may not work in main script e.g. flatten()

process ChaosGameRepresentation {
    publishDir "${params.saveDir}/${directory}/train/",
                mode: params.publish_dir_mode, 
                enabled: params.save,
                pattern: "*train*.tar.gz"
                
    publishDir "${params.saveDir}/${directory}/test/",
                mode: params.publish_dir_mode, 
                enabled: params.save,
                pattern: "*test*.tar.gz"

    // maxForks 3
    label "with_cpus"
    tag   "${directory}"

    input:
        // tuple val(parentDir), path(directory), path(fastaFile)
        path directory

    output:
        tuple  val("${directory}"), path("*train.tar.gz") optional true
        tuple  val("${directory}"), path("*test.tar.gz") optional true
         // Relevant as a reference to the next process in the main script (training/testing)


    script:
    def fastaFile = "Sequences.fasta"

    if (params.trainOnly) {
        """
        cd ${directory}/train/
        chaos_game_representation_of_protein_sequences.R --fasta ${fastaFile} -t
        cd ..
        tar czvf ${directory.baseName}train.tar.gz train/
        """
    } else if (params.testOnly) {
        """
        cd ${directory}/test/
        chaos_game_representation_of_protein_sequences.R --fasta fastaFile -t
        cd ..
        tar czvf ${directory.baseName}test.tar.gz test/
        """
    } else {
        """
        cd ${directory}/train/
        chaos_game_representation_of_protein_sequences.R --fasta fastaFile -t
        cd ..
        tar czvf ${directory.baseName}train.tar.gz -t train/

        cd test/
        chaos_game_representation_of_protein_sequences.R --fasta fastaFile -t
        cd ..
        tar czvf ${directory.baseName}test.tar.gz test/
        """
    }
}

// May add more features
