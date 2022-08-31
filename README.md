# Zoon0PredV

To run full pipeline: _i.e. Data preparation, cleaning, splitting, sequence encoding, training, testing and proof of concept_

N E X T F L O W  >= 22.04.0

```{bash}
nextflow run Rudolph-afk/Zoon0PredV -params-file params/params.yml
```

N E X T F L O W  < 22.04.0

```{bash}
nextflow run Rudolph-afk/Zoon0PredV -params-file params/params.yml -dsl2
```


## Repo Description

| **Directory/File**                      | **Contents**                                                    |
| ---------------------------             | --------------------------------------------------              |
| **[app](app/)**                         | Project dashboard app                                           |
| **[assets](assets/)**                   | Figures and tables                                              |
| **[bin](bin/)**                         | Scripts used in the pipeline                                    |
| **[conf](conf/)**                       | Nextflow pipeline configuration files                           |
| **[data](data/)**                       | Input data for the pipeline                                     |
| **[image-defs](image-defs/)**           | Singularity container definition files                          |
| **[modules](modules/)**                 | Nextflow process definitions                                    |
| **[workflows](workflows/)**             | Nextflow workflow definitions                                   |
| **[notebooks](notebooks/)**             | ipynb notebooks for prototyping and project figure generation   |
| **[run_scripts](run_scripts/)**         | Custom scripts to run the pipeline                              |
| **[params](params/)**                   | Parameter files for the pipeline                                |
| **[main.nf](main.nf/)**                 | Main script                                                     |
| **[nextflow.config](nextflow.config/)** | Main Nextflow configuration file                                |

This project is part of a MSc thesis entitled _A DEEP LEARNING APPROACH TO PREDICTING POTENTIAL VIRUS SPECIES CROSSOVER USING CONVOLUTIONAL NEURAL NETWORKS AND VIRAL PROTEIN SEQUENCE PATTERNS_ submittted to the University of the Western Cape.
