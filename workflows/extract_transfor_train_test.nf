include { LoadCleanData }                   from '../modules/preprocessing'
include { ChaosGameRepresentation }         from '../modules/sequence_encoding'
include { ModelTraining }                   from '../modules/train_conv_net'
include { ModelTestEval }                   from '../modules/test_conv_net'
include { AttentionMapping }                from '../modules/visualize_conv_net'

// Extract, Transform, Load, Train, and Test
workflow ExTrLoad {
    main:
        // Outputs directories (MetazoaData etc.), with accompanying data, as a list channel
        LoadCleanData(
            file(params.prot),
            file(params.ncbiVirus),
            file(params.eID),
            file(params.virusDB),
            file(params.fasta)
            )
        LoadCleanData.out[0]
                    .flatten()
                    .set{ dirSplits }

        LoadCleanData.out[1].set { proof_of_concept_sequences }
        LoadCleanData.out[2].set { proof_of_concept_dataframe }

    emit:
        dirSplits
        proof_of_concept_sequences
        proof_of_concept_dataframe
}

workflow TrainTest {
    take:
        dirSplits

    main:
        ChaosGameRepresentation(dirSplits)
        ChaosGameRepresentation.out[0].tap{ data }

        ModelTraining(data)
        ModelTraining.out[0].set{ models }

        ModelTestEval(models.collect()) // Takes directories with model and test data

    emit:
        models
}

workflow ProofOfConcept {
    take:
        sequences
        models
        dataframe

    main:
        ChaosGameRepresentation(sequences)
        ChaosGameRepresentation.out[1].set{ poc_fcgr_images }

        AttentionMapping(models, poc_fcgr_images, dataframe)
}

workflow Main {
    ExTrLoad()

    TrainTest(
        ExTrLoad.out[0]
    )

    ProofOfConcept(
        ExTrLoad.out[1],
        TrainTest.out,
        ExTrLoad.out[2]
    )
}