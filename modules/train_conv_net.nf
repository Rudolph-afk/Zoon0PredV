
process ModelTraining {
    publishDir  "$params.saveDir",
		        mode:       params.publish_dir_mode,
                enabled:    params.save && !params.trainOnly

    tag         "$directory"
    label       "with_gpus"

    input:
        path    directory

    output:
        path    "$directory"

    script:
        """
        train_zoonosis_model.py -d $directory
        """
}

process SaveModels {
    publishDir  "$params.saveDir",
                mode:       params.publish_dir_mode

    input:
        path    fileFolderName

    output:
        path    "$fileFolderName"

    script:
        """
        echo 'Saving $fileFolderName'
        """
}