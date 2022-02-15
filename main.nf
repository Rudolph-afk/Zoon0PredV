<<<<<<< HEAD
#!/usr/bin/env nextflow
=======
#!/bin/nextflow

nextflow.enable.dsl=2
>>>>>>> 6377fef3073d3822750dc9f1941ae577b02fdca9

include { PrepData } from './modules/extract_transform'
include { ChaosGameRepresentation } from './modules/feature_extraction'
include { ModelTraining } from './modules/train_conv_net'


// uniprot     = Channel.fromPath(params.prot,      type: 'file', glob: false)
// ncbiVirus   = Channel.fromPath(params.ncbiVirus, type: 'file', glob: false)
// eid         = Channel.fromPath(params.eID,       type: 'file', glob: false)
// virus_db    = Channel.fromPath(params.virusDB,   type: 'file', glob: false)
// fasta       = Channel.fromPath(params.fasta,     type: 'file', glob: false)


workflow {
    // Outputs directories (MetazoaData etc.) with accompanying data as a list
    PrepData(
            file(params.prot),
            file(params.ncbiVirus), 
            file(params.eID), 
            file(params.virusDB), 
            file(params.fasta)
        ).out
        .flatten()
        .set{ dirSplits } 

    // Outputs tuples in the form (parentDirectory, subDirectory, FASTAFile)
<<<<<<< HEAD
    // fcgrTestSplit  = dirSplits
    //                     .flatten()
    //                     .map { it -> def split=it; tuple(split, file("${split}/test"), file("${split}/test/*.fasta")) }
    // fcgrTrainSplit = dirSplits
    //                     .flatten()
    //                     .map { it -> def split=it; tuple(split, file("${split}/train"), file("${split}/train/*.fasta")) }
=======
    fcgrTrainSplit = dirSplits
                        .flatten()
                        .map {
                            it -> def split=it.baseName;
                            tuple(
                                split, 
                                file("${params.saveDir}/${split}/train"),
                                file("${params.saveDir}/${split}/train/Sequences.fasta")
                            )
                        }
>>>>>>> 6377fef3073d3822750dc9f1941ae577b02fdca9
    
    if (params.test == true) {
        fcgrTestSplit  = dirSplits
                            .flatten()
                            .map {
                                it -> def split=it.baseName;
                                tuple(
                                    split,
                                    file("${params.saveDir}/${split}/test"),
                                    file("${params.saveDir}/${split}/test/Sequences.fasta")
                                )
                            }

    // Combines the 2 channels to output a single channel emitting the tuples
<<<<<<< HEAD
    // fCGRData = fcgrTrainSplit
    //                 .mix(fcgrTestSplit)

    // Outputs tuples in the form (parentDirectory, subDirectory)
    // fCGR = ChaosGameRepresentation(fCGRData) // fix from here

    ChaosGameRepresentation(dirSplits).out
                                        .set{ data }

    // fCGR = fCGR.map { it[0] } // Flatten will try to flatten individual output and will not produce desired output
    // data = fCGR = fCGR.unique()
    //                     .map { it -> def dir=it; file(dir) }
    // fCGRmoved = MoveCGRImages(fCGR)
    
    // data = fCGRmoved.flatten()
    //                 .map { it -> def dir=it; file(dir) }
    if ( params.trainOnly ) {
        ModelTraining(data)
    } // else if ( params.testOnly ) {
        // TestModel
    } else {
        ModelTraining(data)
        // TestModel
    }
=======
        fCGRData = fcgrTrainSplit
                    .mix(fcgrTestSplit)
    } else {
        fCGRData = fcgrTrainSplit
    }

    // Outputs tuples in the form (parentDirectory, subDirectory)
    ChaosGameRepresentation(fCGRData) // fix from here

    // fCGR = fCGR.map { it[0] } // Flatten will try to flatten individual output and will not produce desired output
    ChaosGameRepresentation.out[1].flatten()
                            .unique()
                            .set{ data }
                            // .map { it -> def dir=it; path(dir) }
    // fCGRmoved = MoveCGRImages(fCGR)
    
    data = data
            .map { it -> def dir=it; file("${params.saveDir}/${dir}") }

    ModelTraining(data)
>>>>>>> 6377fef3073d3822750dc9f1941ae577b02fdca9
}

// workflow {
//     main:
//         prepareData()

//         // Outputs directories alongside their added models subdirectory
//          // Saves to specified directory if params.save is true

//         // Workflow is complete
// }
