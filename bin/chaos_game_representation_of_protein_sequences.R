#!/opt/conda/bin/Rscript --vanilla
args = commandArgs(trailingOnly=TRUE)
suppressPackageStartupMessages({
  library('argparse')
  library('kaos')
  library('protr')
  library('stringr')
  library('progress')
  library('parallel')
  library('foreach')
  library('doParallel')
})

registerDoParallel(cores=8)

parser <- ArgumentParser(description='Create FCGR of FASTA sequences')

parser$add_argument('--fasta', help='Fasta file')
parser$add_argument('--directory', help='Save directory')

args <- parser$parse_args()

file = args$fasta
# dir = str_split(file, '/', simplify = T)
# directory = dir[1:(length(dir)-1)]
directory = args$directory

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
names(v.seqs) = seq.names

print(names(v.seqs)[1])

chaos_game_representation = function (x) {
  chaos.obj = cgr(str_split(x, '', simplify = T))
  chaos.plot = cgr.plot(chaos.obj, mode = "matrix", corners=T, labels=T)
  return(chaos.plot)
}

# pb <- progress_bar$new(
#     format = "(:spin)  Processing [:bar] :percent in :elapsed",
#     total = length(v.seqs), clear = FALSE, width= 60)

# print("Starting CGR")

foreach(sequence=v.seqs, seq.name=seq.names) %dopar% {
    chaos.graph.objs = chaos_game_representation(sequence)
    name = unlist(seq.name)
    name = paste(directory, name, sep="/")
    file_name = paste(name, ".jpeg", sep="")
    jpeg(file_name, width = 96, height = 96)
    print(chaos.graph.objs)
    dev.off()
    # pb$tick()
}

# save_chaos_game_representation = function (sequence, seq.name) {
#     chaos.graph.objs = chaos_game_representation(sequence)
#     name = unlist(seq.name)
#     name = paste(directory, name, sep="/")
#     file_name = paste(name, ".jpeg", sep="")
#     jpeg(file_name, width = 96, height = 96)
#     print(chaos.graph.objs)
#     dev.off()
# }

# mapply(save_chaos_game_representation, v.seqs, seq.names)