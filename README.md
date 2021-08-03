# Zoon Pred Model

| **Directory** | **Contents**                                       |
|---------------|----------------------------------------------------|
| **bin**       | R and python scripts                               |
| **modules**   | Nextflow process definitions                       |
| **notebooks** | ipynb notebooks                                    |
| **conf**      | Additional pipeline config files                   |
| **viz**       | Visualizations generated from pipeline             |
| **data**      | Input data and output data generated from pipeline |

## bin
### chaos_game_representation_of_protein_sequences.R
    * Libraries
        - argparse
        - kaos
        - protr
        - stringr
        - parallel
        - foreach
        - doParallel
    * Inputs
        - FASTA file
        - Directory to save the generated CGR images
    * Outputs
        - cgr images prefixed with human_true or human_false 

### data_cleaning.py
    * Inputs
        - Uniprot tab file
        - NCBI Virus csv file
        - EID2 from the university of Liverpool csv file
        - virus database csv file
        - Uniprot fasta file
    * Outputs
        - 6 Directories containing output csv files and subdirectories (train, test)

### group_proteins.py
**incomplete & not part of pipeline**

**will use in dash-app for MSA viewing**

    * Inputs
        - FASTA sequence
    * Outputs
        - multiple fasta files with sequences grouped based on their names

### hyperparameter_search.py 
**incomplete, will convert BayesianHparams notebook to script**

    * Input 
        - Directory caontaining human-true, human-false subdirectories containing cgr images
    * output
        - logs directory and summary of top 10 hyperparams combinations 

### test_zoonosis_model.py
**incomplete, will convert Results notebook to script**

    * Input
        - test directories
        - models from the various directories
    * Output
        - Plots showing model metrics into viz directory

### train_zoonosis_model.py

    * Inputs
        - Base directory (assumes base directory has train subdirectory with cgr data)
        - other arguments are not required unless the train data is to be obtained from a different directory
    * Outputs
        - checkpoint files
        - directory named model containing saved model
        - training logs in csv

### zoonosis_helper_functions.py
    * helper functions using in the data cleaning process

## modules
### extract_transform.nf
    * process running the data_clean.py script

### feature_extraction.nf
    * process running the chaos_game_representation_of_protein_sequences.R script

### hyperparameter_search.nf
**incomplete**
    * process running the hyperparameter_search.py script 

### proteins_pipeline.nf
**used for dash-app visualisation**
    * multiple processes processing the proteins FASTA files 

### train_conv_net.nf
    * process running the train_zoonosis_model.py script

## notebooks
### Zoonosis_data_cleaning.ipynb
    * Data cleaning notebook, converted to data_cleaning.py script

### zoonosis_model.ipynb
    * Single zoonosis model, training. Modified and converted to train_zoonosis_model.py

### Results.ipynb
    * Test and metrics plots, to be converted to test_zoonosis_model.py

### R_notebook.ipynb
    * Shows a bit of the CGR code and CGR image

### BayesianHyperparameterSearch.ipynb
    * hyperparameter search process using Bayesian Hyperparameter search method

## data
### sequences.csv
    * NCBI virus csv file with virus name, host name, molecule type

### uniprot-keyword Virus+entry+into+host+cell+\[KW-1160\] +fragment no.fasta
    * Uniprot data containing FASTA sequence corresponding to the uniprot tab file

### uniprot-keyword Virus+entry+into+host+cell+\[KW-1160\] +fragment no.tab.gz
    * Uniprot data containing virus protein entries 

### virus_host_4rm_untitled.csv
    * EID2 data from the University of Liverpool

### virushostdb.tsv
    * Virus host database data
