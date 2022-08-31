#!/usr/local/bin/python

import argparse
import os
import pandas as pd


def get_hyper_from_json(directory):
    file = os.path.join(directory, 'trial.json')
    json_df = pd.read_json(file)
    hparams = pd.Series(json_df.hyperparameters[1])
    metrics = (pd.Series(json_df.metrics[2])
               .apply(
                   lambda metrics: metrics['observations'][0]['value'][0]))
    return metrics.append(hparams).to_frame().T


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Zoonosis Model Evaluation',
                                     description=\
                                     """
                                     Get best hyperparameters and score
                                     """.strip(),
                                     )

    parser.add_argument('-d', '--directory',
                        required=True,
                        help='Main directory containing keras tuner trial oracles')

    args = parser.parse_args()
    
    work_dir = args.directory
    
    directories = list(filter(os.path.isdir, [os.path.join(work_dir, directory) for directory in os.listdir(work_dir)]))

    df = pd.concat(list(map(get_hyper_from_json, directories)), axis=0)


    df = df[[
        'units',
        "kernel",
        "pool",
        "optimizer",
        "threshold",
        "learning_rate",
        "model_layers",
        "val_MCC"
    ]].sort_values('val_MCC', ascending=False)


    df.to_csv("hyperparameters.csv", index=False)