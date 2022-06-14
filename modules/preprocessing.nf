
process LoadCleanData {
    publishDir  "$params.saveDir/Results/$workflow.runName/",
                mode:       params.publish_dir_mode,
                enabled:    params.save

    label       "with_cpus"
    tag         "Data Cleaning"

    input:
        path    prot
        path    ncbiVirus
        path    eID
        path    virusDB
        path    fasta
        // path    taxonomy_database

    output:
        path    "*", type: "dir"
        path    "*.fasta"
        path    "*.csv"

    script:
        """
        data_cleaning.py --uniprot $prot \
                         --ncbivirusdb $ncbiVirus \
                         --liverpooluni $eID \
                         --virusdb $virusDB \
                         --fasta $fasta
        """
}

process ExtractFiles {
    tag         "Extracting"

    input:
        path    compressed_file

    output:
        path    "*prot"
        path    "*ncbiVirus"
        path    "*eID"
        path    "*virusDB"
        path    "*fasta"
        path    "*.tar.gz"

    script:
        """
        tar xjf $compressed_file
        """
}