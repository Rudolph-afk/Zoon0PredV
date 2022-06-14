
process ChaosGameRepresentation {

    label       "with_cpus"
    tag         "$directory"

    input:
        path    directory

    output:
        path    "$directory"
        path    "*.png" optional true

    script:
        def train        = "${directory}/train/"
        def test         = "${directory}/test/"
        switch (directory) {
            case { directory.isFile() }:
                """
                chaos_game_representation_of_protein_sequences.R --fasta Sequences.fasta
                """
                break;

            default:
                if (params.type == "ExtractTrain") {
                    runChaos(train)
                } else if (params.type == "ExtractTest") {
                    runChaos(test)
                } else {
                    runChaos(train) + "\ncd ../..\n" + runChaos(test)
                }
        }
}

def runChaos(dataDir) {
    """
    cd $dataDir
    chaos_game_representation_of_protein_sequences.R --fasta Sequences.fasta -p
    """
}