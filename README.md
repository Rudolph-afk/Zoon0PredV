# Zoon0PredV

## Execution

To run full pipeline: Data preparation, features extractiona, training and testing

    ```{bash}
    nextflow run main.nf -dsl2
    ```

To run pipeline to just train: Data preparation, feature extraction, and training (no testing)

    ```{bash}
    nextflow run main.nf -dsl2
                --trainOnly
    ```

To run pipeline to just test -- Feature extraction and testing (Assumes the model already exists)

    ```{bash}
    nextflow run main.nf -dsl2
                --testOnly
    ```

## Repo Description

| **Directory**               | **Contents**                                       |
| --------------------------- | -------------------------------------------------- |
| **[app](app/)**             | Project dashboard app                              |
| **[bin](bin/)**             | R and python scripts                               |
| **[conf](conf/)**           | Additional pipeline config files                   |
| **[data](data/)**           | Input data and output data generated from pipeline |
| **[modules](modules/)**     | Nextflow process definitions                       |
| **[notebooks](notebooks/)** | ipynb notebooks                                    |

### app

**[appHelperFuncs.py](app/appHelperFuncs.py)**

    * Helper functions used in theh dashboad app

**[dash_app.py](app/dash_app.py)**

    * Zoon0PredV dashbaord app

### bin

**[chaos_game_representation_of_protein_sequences.R](bin/chaos_game_representation_of_protein_sequences.R)**

    * Inputs
        - FASTA file
        - Directory to save the generated CGR images
    * Outputs
        - cgr images prefixed with human_true or human_false

**[data_cleaning.py](bin/data_cleaning.py)**

    * Inputs
        - Uniprot tab file
        - NCBI Virus csv file
        - EID2 from the university of Liverpool csv file
        - virus database csv file
        - Uniprot fasta file
    * Outputs
        - 6 Directories containing output csv files and subdirectories (train, test)

**[group_proteins.py](bin/group_proteins.py)**

_**incomplete & not part of pipeline**_

_**will use in dash-app for MSA viewing**_

    * Inputs
        - FASTA sequence
    * Outputs
        - multiple fasta files with sequences grouped based on their names

**[hyperparameter_search.py](bin/hyperparameter_search.py)**

_**incomplete, will convert BayesianHparams notebook to script**_

    * Input
        - Directory caontaining human-true, human-false subdirectories containing cgr images
    * output
        - logs directory and summary of top 10 hyperparams combinations

**[test_zoonosis_model.py](bin/test_zoonosis_model.py)**

_**incomplete, will convert Results notebook to script**_

    * Input
        - test directories
        - models from the various directories
    * Output
        - Plots showing model metrics into viz directory

**[train_zoonosis_model.py](bin/train_zoonosis_model.py)**

    * Inputs
        - Base directory (assumes base directory has train subdirectory with cgr data)
        - other arguments are not required unless the train data is to be obtained from a different directory
    * Outputs
        - checkpoint files
        - directory named model containing saved model
        - training logs in csv

**[zoonosis_helper_functions.py](bin/zoonosis_helper_functions.py)**

    * helper functions using in the data cleaning process

### modules

**[extract_transform.nf](modules/extract_transform.nf)**

    * process running the data_clean.py script

**[feature_extraction.nf](modules/feature_extraction.nf)**

    * process running the chaos_game_representation_of_protein_sequences.R script

**[hyperparameter_search.nf](modules/hyperparameter_search.nf)**

_**incomplete**_

    * process running the hyperparameter_search.py script

**[proteins_pipeline.nf](modules/proteins_pipeline.nf)**

_**used for dash-app visualisation**_

    * multiple processes processing the proteins FASTA files

**[train_conv_net.nf](modules/train_conv_net.nf)**

    * process running the train_zoonosis_model.py script

### notebooks

**[Zoonosis_data_cleaning.ipynb](notebooks/Zoonosis_data_cleaning.ipynb)**

    * Data cleaning notebook, converted to data_cleaning.py script

**[zoonosis_model.ipynb](notebooks/zoonosis_model.ipynb)**

    * Single zoonosis model, training. Modified and converted to train_zoonosis_model.py

**[Results.ipynb](notebooks/Results.ipynb)**

    * Test and metrics plots, to be converted to test_zoonosis_model.py

**[BayesianHyperparameterSearch.ipynb](notebooks/BayesianHyperparameterSearch.ipynb)**

    * Hyperparameter search process using Bayesian Hyperparameter search method

**[chaosGR.Rmd](notebooks/chaosGR.Rmd)**

    * Chaos game representation code, loop variant and the async variant and benchmark speeds

**[plots.png](notebooks/plots.png)**

    * Image of CGR of 4 selected viral proteins

**[zoonosis_helper_functions.py](notebooks/zoonosis_helper_functions.py)**

    * Same script as in [bin directory](bin/zoonosis_helper_functions.py)

### data

**[sequences.csv](data/sequences.csv)**

    * NCBI virus csv file with virus name, host name, molecule type

**[uniprot-keyword Virus+entry+into+host+cell+\[KW-1160\] +fragment no.fasta](data/uniprot-keyword Virus+entry+into+host+cell+\[KW-1160\] +fragment no.fasta)**

    * Uniprot data containing FASTA sequence corresponding to the uniprot tab file

**[uniprot-keyword Virus+entry+into+host+cell+\[KW-1160\] +fragment no.tab.gz](data/uniprot-keyword Virus+entry+into+host+cell+\[KW-1160\] +fragment no.tab.gz)**

    * Uniprot data containing virus protein entries

**[virus_host_4rm_untitled.csv](data/virus_host_4rm_untitled.csv)**

    * EID2 data from the University of Liverpool

**[virushostdb.tsv](data/virushostdb.tsv)**

    * Virus host database data

## Conf

**[base.config](conf/base.config)**

    * Nextflow additional configuration file for running the pipeline on a local machine

**[ilifu-cluster.config](conf/ilifu-cluster.config)**

    * Nextflow additional configuration file for running the pipeline on the ilifu cluster on a GPU node

**[ilifu.config](conf/ilifu.config)**

    * Nextflow additional configuration file for running the pipeline on the ilifu cluster

**[sanbi-cluster.config](conf/sanbi-cluster.config)**

    * Nextflow additional configuration file for running the pipeline on the sanbi cluster
