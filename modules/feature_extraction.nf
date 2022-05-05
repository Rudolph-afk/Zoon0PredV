nextflow.enable.dsl=2 // Required as some methods/functions may not work in main script e.g. flatten()

process ChaosGameRepresentation {
    // publishDir  "$params.saveDir/$directory/",
    //             mode:    params.publish_dir_mode,
    //             enabled: params.save && !params.testOnly && !params.trainOnly,
    //             saveAs:  { filename =~ /train/ ? "train/train.tar.bz2" : "test/test.tar.bz2" },
    //             pattern: "*Data/**/*.tar.bz2"

    label       "with_cpus"
    tag         "$directory"

    input:
        path    directory

    output:
        path    "$directory" // optional (params.testOnly && params.trainOnly)

    script:
        def train        = "$directory/train/"
        def test         = "$directory/test/"

        if (params.trainOnly) {
            runChaos(train)
        } else if (params.testOnly) {
            runChaos(test)
        } else {
            runChaos(train) + "\ncd ../..\n" + runChaos(test)
        }
}

def runChaos(dataDir) {
    def split = dataDir =~ /train/ ? 'train' : 'test';

    """
    cd $dataDir
    chaos_game_representation_of_protein_sequences.R --fasta Sequences.fasta -p
    """
}