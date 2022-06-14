
process HyperParameterSearch {
    publishDir  "$params.saveDir/Results/",
                mode:  params.publish_dir_mode

    label       "with_gpus"

    input:
        path    data

    output:
        path    "*.{csv,png}"

    script:
        """
        hyperparameter_search.py -d $data/train
        """
}
