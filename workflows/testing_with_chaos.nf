include { ChaosGameRepresentation }         from '../modules/sequence_encoding'
include { ModelTestEval }                   from '../modules/test_conv_net'

workflow ChaosTestEval {
    Channel.fromPath(params.data, type: "dir")
            .set{ test }

    ChaosGameRepresentation(test)
    ChaosGameRepresentation.out[0]
                            .collect()
                            .set{ data }

    ModelTestEval(data)
}