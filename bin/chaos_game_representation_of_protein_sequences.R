#!/opt/conda/bin/Rscript --vanilla
args = commandArgs(trailingOnly=TRUE)
suppressPackageStartupMessages({
    library('argparse')
    library('kaos')
    library('protr')
    library('stringr')
    library('progress') # Not used in pipeline but usefull when running directly in terminal
    library('parallel')
    library('later')
    library('foreach') # Not required by main implementation
    library('doParallel') # Also not req.
})

n.cores <- detectCores()

parser <- ArgumentParser(description='Create FCGR of FASTA sequences')

parser$add_argument('--fasta', help='Fasta file')

args <- parser$parse_args()

file = args$fasta

v.seqs = unlist(readFASTA(file = file))

print(length(v.seqs))

remove_slash = function (x) {
    x = gsub("[[:space:]]", "_", x, fixed = TRUE)
    x = gsub("|", "_", x, fixed = TRUE)
    x = gsub("=", "_", x, fixed = TRUE)
    x = gsub("-", "_", x, fixed = TRUE)
    result = gsub("/", "_", x, fixed = TRUE)
    return(result)
}

seq.names = names(v.seqs)
seq.names = unlist(seq.names)
seq.names = lapply(seq.names, remove_slash)

specify_directory <- function(x) {
    if (grepl(pattern = "true", x = x)) {
        new_name <- paste("human-true", x, sep="/")
    } else {
        new_name <- paste("human-false", x, sep="/")
    }
    return(new_name)
}

seq.names = lapply(seq.names, specify_directory)

names(v.seqs) = seq.names

chaos_game_representation = function (x) {
    chaos.obj = cgr(str_split(x, '', simplify = T))
    chaos.plot = with_temp_loop(
      cgr.plot(chaos.obj, mode = "matrix", corners=T, labels=T)
      )
    return(chaos.plot)
}

save_chaos_game_representation <- function (sequence, seq.name) {
    file_name <- paste(seq.name, ".jpeg", sep="")
    chaos.graph <- chaos_game_representation(sequence)
    with_temp_loop({
      jpeg(file_name, width = 96, height = 96)
      print(chaos.graph)
      dev.off()
      }
    )
}

######################### Main #######################################

loop = global_loop()

with_loop(
    loop,
    mcmapply(
        save_chaos_game_representation,
        sequence = v.seqs,
        seq.name = seq.names,
        mc.cores = n.cores
    )
  )

####################### exmple implementation of progress bar ############
# pb <- progress_bar$new(
#     format = "(:spin)  Processing [:bar] :percent in :elapsed",
#     total = length(v.seqs), clear = FALSE, width= 60)

# print("Starting CGR")

############################### SLOW #####################################
# foreach(sequence=v.seqs, seq.name=seq.names) %dopar% {
#     chaos.graph = chaos_game_representation(sequence)
#     name = unlist(seq.name)
#     # name = paste(directory, name, sep="/")
#     file_name = paste(name, ".jpeg", sep="")
#     jpeg(file_name, width = 96, height = 96)
#     print(chaos.graph)
#     dev.off()
#     # pb$tick()
# }

############################## VERY SLOW ################################
# save_chaos_game_representation = function (sequence, seq.name) {
#     chaos.graph = chaos_game_representation(sequence)
#     name = unlist(seq.name)
#     name = paste(directory, name, sep="/")
#     file_name = paste(name, ".jpeg", sep="")
#     jpeg(file_name, width = 96, height = 96)
#     print(chaos.graph)
#     dev.off()
# }
# mapply(save_chaos_game_representation, v.seqs, seq.names)
