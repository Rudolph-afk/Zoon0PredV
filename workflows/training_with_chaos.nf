include { ChaosGameRepresentation }         from '../modules/sequence_encoding'
include { ModelTraining }                   from '../modules/train_conv_net'

workflow ChaosTrain {
    Channel.fromPath(params.data, type: "dir")
            .set{ train }

    ChaosGameRepresentation(train)
    ChaosGameRepresentation.out[0].set{ data }

    ModelTraining(data)
}