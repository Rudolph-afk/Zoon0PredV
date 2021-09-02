#!/usr/bin/env nextflow
nextflow.enable.dsl=2

include { PrepData } from './modules/extract_transform'
include { ChaosGameRepresentation; MoveCGRImages } from './modules/feature_extraction'
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
    fcgrTestSplit  = dirSplits
                        .flatten()
                        .map { it -> def split=it; tuple(split, file("${split}/test"), file("${split}/test/*.fasta")) }
    fcgrTrainSplit = dirSplits
                        .flatten()
                        .map { it -> def split=it; tuple(split, file("${split}/train"), file("${split}/train/*.fasta")) }
    
    // splits_as_vals = dirSplits.flatten().map { it -> def split=it; val(split) }

    // Combines the 2 channels to output a single channel emitting the tuples
    fCGRData = fcgrTrainSplit
                    .mix(fcgrTestSplit)

    // Outputs tuples in the form (parentDirectory, subDirectory)
    fCGR = ChaosGameRepresentation(fCGRData) // fix from here

    // fCGR = fCGR.map { it[0] } // Flatten will try to flatten individual output and will not produce desired output
    data = fCGR = fCGR.unique()
                        .map { it -> def dir=it; file(dir) }
    // fCGRmoved = MoveCGRImages(fCGR)
    
    // data = fCGRmoved.flatten()
    //                 .map { it -> def dir=it; file(dir) }
    ModelTraining(data)
}

// workflow {
//     main:
//         prepareData()

//         // Outputs directories alongside their added models subdirectory
//          // Saves to specified directory if params.save is true

//         // Workflow is complete
// }
