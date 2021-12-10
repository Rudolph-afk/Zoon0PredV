#!/bin/nextflow

nextflow.enable.dsl=2

include { PrepData } from './modules/extract_transform'
include { ChaosGameRepresentation } from './modules/feature_extraction'
include { ModelTraining } from './modules/train_conv_net'


uniprot     = Channel.fromPath(params.prot,      type: 'file', glob: false)
ncbiVirus   = Channel.fromPath(params.ncbiVirus, type: 'file', glob: false)
eid         = Channel.fromPath(params.eID,       type: 'file', glob: false)
virus_db    = Channel.fromPath(params.virusDB,   type: 'file', glob: false)
fasta       = Channel.fromPath(params.fasta,     type: 'file', glob: false)


workflow {
    // Outputs directories (MetazoaData etc.) with accompanying data as a list
    dirSplits = PrepData(uniprot, ncbiVirus, eid, virus_db, fasta) 

    // Outputs tuples in the form (parentDirectory, subDirectory, FASTAFile)
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
}

// workflow {
//     main:
//         prepareData()

//         // Outputs directories alongside their added models subdirectory
//          // Saves to specified directory if params.save is true

//         // Workflow is complete
// }
