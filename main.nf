#!/usr/bin/env nextflow

include { PrepData } from './modules/extract_transform'
include { ChaosGameRepresentation } from './modules/feature_extraction'
include { ModelTraining } from './modules/train_conv_net'


workflow ExTrLoadTT { // Extract Transform Load Train and Test
    // Outputs directories (MetazoaData etc.) with accompanying data as a list channel
    PrepData(
            file(params.prot),
            file(params.ncbiVirus), 
            file(params.eID), 
            file(params.virusDB), 
            file(params.fasta)
        ).out
        .flatten()
        .set{ dirSplits } 


    ChaosGameRepresentation(dirSplits).out
                                        .set{ data }
    ModelTraining(data).out[0].set{ model }
    model
        .combine(data, by: 0)
        .multiMap{
            model: it -> tuple(it[0], it[1])
            data:  it -> tuple(it[0], it[2])
        }
        .set{ result }
    // ModelTesting(model, data)
}

workflow {
    if ( params.trainOnly ) {
        Channel.fromPath(params.train_data_tarball).set{ data }
        ModelTraining(data)
    } else if ( params.testOnly ) {
        // Channel.fromPath(test_data)
        // .multiMap{
        //     model: it -> tuple(val(it.baseName), file("${it}/model/"))
        //     data:  it -> tuple(val(it.baseName), file("${it}/test/test.tar.gz"))
        //     }
        // .set{ result }
        // ModelTesting(result.model, result.data)
    } else {
        ETL_T_T()
    }
}
// workflow {
//     main:
//         prepareData()

//         // Outputs directories alongside their added models subdirectory
//          // Saves to specified directory if params.save is true

//         // Workflow is complete
// }
