#!/usr/bin/Rscript --vanilla

# install.packages(c('argparse', 'kaos','protr', 'bookdown',
#                   'stringr', 'progress', 'parallel', 'knitr',
#                   'foreach', 'doParallel', 'rmarkdown'
#                   ),
#            dependencies=TRUE, repos='http://cran.rstudio.com/')

install.packages(c("reticulate", "WGCNA", "DESeq2"),
            dependencies=TRUE,
            repos='http://cran.rstudio.com/')
