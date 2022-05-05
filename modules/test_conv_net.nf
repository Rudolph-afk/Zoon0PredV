
process ModelTestEval {
    publishDir  "$params.saveDir/Results",
                mode:  params.publish_dir_mode

    label       "with_gpus"

    input:
        path    directories //, stageAs: "${directories.baseName}"

    output:
        path    "{*.csv,*.png}"
        // path    "${directories.each{it.baseName}}" // optional params.testOnly

    script:
        """
        test_zoonosis_model.py -d .
        """
}


// process ModelResults {
//     publishDir  "$params.saveDir/Results",
//                 mode:       params.publish_dir_mode,
//                 enabled:    params.save && !params.testOnly && !params.trainOnly

//     label       "with_gpus" // For using the

//     input:
//         path    evals

//     output:
//         path "*.png"

//     script:
//         """
//         results.py -d
//         """
// }