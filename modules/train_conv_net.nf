
process ModelTraining {
    publishDir  "$params.saveDir/Results/$workflow.runName/$directory",
		        mode:       params.publish_dir_mode,
                enabled:    params.save,
                pattern:    "*.csv"

    tag         "$directory"
    label       "with_gpus"

    input:
        path    directory

    output:
        path    "$directory"
        path    "*.csv"
        // path    "$directory/model"

    script:
        """
        train_zoonosis_model.py -d $directory
        """
}
