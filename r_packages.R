#!/opt/conda/bin/Rscript --vanilla

install.packages(c('argparse', 'kaos','protr', 'later',
                   'stringr', 'progress', 'parallel',
                   'foreach', 'doParallel'), dependencies=TRUE, repos='http://cran.rstudio.com/')
