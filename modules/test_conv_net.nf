
process ModelTestEval {
    publishDir  "$params.saveDir/Results/$workflow.runName",
                mode:  params.publish_dir_mode

    label       "with_cpus"

    input:
        path    directories //, stageAs: "${directories.baseName}"

    output:
        path    "*.{csv,png}"
        // path    "${directories.each{it.baseName}}" // optional params.testOnly

    script:
        """
        test_zoonosis_model.py -d .
        """
}
