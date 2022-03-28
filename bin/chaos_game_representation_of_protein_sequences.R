#!/usr/bin/Rscript --vanilla

# args = commandArgs(trailingOnly=TRUE)
suppressPackageStartupMessages({
    library('argparse') # Requires active python installation
    library('kaos') # Chaos game representation
    library('protr') # Reading FASTA files
    library('stringr')
    library('progress') # Not used in pipeline but useful for development
    library('parallel')
    library('later') # Implementing asynchronous programming
    library('foreach') # Not required by main implementation
    library('doParallel') # Also not req.
})

n.cores <- detectCores()

parser <- ArgumentParser(description='Create a FCGR representation of FASTA protein sequences')

parser$add_argument('--fasta', help='File - Fasta file')
parser$add_argument('-t', dest="prefix", action='store_true',
                    help='sequence ID contains "human_true" or "human_false" as prefix. If so save image in directory named "human-true" or "human-false", respectively (default=FALSE')
parser$add_argument('--label', '-l', action='store_true', help='add corners and labels to FCGR (default=FALSE)')
parser$add_argument('--size', '-s', type="integer", default=600, help='Integer - width and height of the image, image width is equal to height (default: width=height=600)')

args <- parser$parse_args()

file = args$fasta

prefix.human = args$prefix

CORNERS.LABELS = args$label

WIDTH.HEIGHT = args$size

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

specify_directory <- function(x) {
  if (grepl(pattern = "true", x = x)) {
    new_name <- paste("human-true", x, sep="/")
  } else {
    new_name <- paste("human-false", x, sep="/")
  }
  return(new_name)
}

seq.names = names(v.seqs)
seq.names = unlist(seq.names)
seq.names = lapply(seq.names, remove_slash)

if (prefix.human){
  seq.names = lapply(seq.names, specify_directory)
}

names(v.seqs) = seq.names

chaos_game_representation = function (x) {
    protein.sequence = str_split(x, '', simplify = TRUE)
    chaos.obj = cgr(protein.sequence, res=300)
    chaos.plot = with_temp_loop(
      cgr.plot(chaos.obj, mode = "matrix", corners=CORNERS.LABELS, labels=T)
      )
    return(chaos.plot)
}

save_chaos_game_representation <- function (sequence, seq.name) {
    file_name <- paste(seq.name, ".png", sep="")
    chaos.graph <- chaos_game_representation(sequence)
    with_temp_loop({
    # JPG performs better for photorealistic images, PNG for drawings with sharp lines and solid colors
      png(file_name, width = WIDTH.HEIGHT, height = WIDTH.HEIGHT)
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
