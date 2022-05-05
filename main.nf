include { LoadCleanData }                   from './modules/extract_transform'
include { ChaosGameRepresentation }         from './modules/feature_extraction'
include { ModelTraining }                   from './modules/train_conv_net'
include { ModelTestEval }                   from './modules/test_conv_net'

// Extract, Transform, Load, Train, and Test
workflow ExTrLoadTT {
    // Outputs directories (MetazoaData etc.), with accompanying data, as a list channel
    LoadCleanData(
        file(params.prot),
        file(params.ncbiVirus),
        file(params.eID),
        file(params.virusDB),
        file(params.fasta)
        )
    LoadCleanData.out
                .flatten()
                .set{ dirSplits }

    ChaosGameRepresentation(dirSplits)
    ChaosGameRepresentation.out.tap{ data }

    ModelTraining(data)
    ModelTraining.out.collect().set{ models }

    ModelTestEval(models) // Takes directories with model and test data
}

workflow {
    if ( params.trainOnly ) {
        Channel.fromPath(params.train_data, type: "dir")
                .set{ train }

        ChaosGameRepresentation(train)
        ChaosGameRepresentation.out.set{ data }

        ModelTraining(data) // Takes directories which contain train data

    } else if ( params.testOnly ) {
        Channel.fromPath(params.test_data, type: "dir")
               .set{ test }

        ChaosGameRepresentation(test)
        ChaosGameRepresentation.out
                                .collect()
                                .set{ data }

        ModelTestEval(data) // Takes directories which contain model and test data
    } else {
        ExTrLoadTT()
    }
}
