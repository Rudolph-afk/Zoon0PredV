#!/usr/local/bin/python

import os
import argparse
# from itertools import cycle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.metrics import  BinaryAccuracy, FalsePositives, TruePositives, TrueNegatives, \
#                                       FalseNegatives, AUC , Recall, Precision
from sklearn.metrics import precision_recall_curve, f1_score, roc_curve
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from model_definition import load_compile_model
from metrics_helper import MatthewsCorrelationCoefficient

############ Activate GPU devices and enable memory growth #################

physical_devices = tf.config.experimental.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, enable=True)

####################### Define functions ##################################

def load_test_data(directory: str):
    data_generator = ImageDataGenerator(rescale=1./255)
    test_data = data_generator.flow_from_directory(
        directory,
        target_size=(224, 224),
        color_mode='grayscale',
        classes=['human-false', 'human-true'],
        class_mode='binary',
        batch_size=2048,
        shuffle=False,
        seed=19980603
    )
    return test_data

def test_model(model, folder: str):

    model_name = folder.split('/')[-2]

    test = load_test_data(folder) # os.path.join(folder, 'test'))

    steps = int(round(test.n / 2048))

    predictions = (model.predict(test, verbose=0)).astype(np.float32).reshape((-1,)).tolist()

    test.reset()
    evals = model.evaluate(test, verbose=0, return_dict=True, steps=steps)

    test.reset()
    true_values = test.labels.astype(np.float32).reshape((-1,)).tolist()

    # Creating dataframe automatically sorts out issue of dimentions of numpy arrays
    df = pd.DataFrame.from_dict({
        'True values':true_values,
        'Predictions':predictions})

    precision, recall, _ = precision_recall_curve(df['True values'], df['Predictions'])
    false_pos, true_pos, _ = roc_curve(df['True values'], df['Predictions'])
    f1 = 2*(evals['Recall'] * evals['Precision']) / (evals['Recall'] + evals['Precision'])

    records = [
        evals["binary_accuracy"], evals["TP"], evals["FP"],
        evals["TN"], evals["FN"], evals["MCC"],
        f1, evals["AUC"], model_name
        ]
    columns = [
        "Accuracy", "True Positive", "False Positive",
        "True Negative", "False Negative", "MCC",
        "F1 Score", "ROC AUC", "Name"
        ]

    metrics_frame = pd.DataFrame([dict(zip(columns, records))])

    PR_curve = {f"Precision_{model_name}": precision.tolist(),
                f"Recall_{model_name}": recall.tolist()}

    ROC_curve = {f"FPR_{model_name}": false_pos.tolist(),
                 f"TPR_{model_name}": true_pos.tolist()}

    return metrics_frame, PR_curve, ROC_curve

line_width = lambda name: 1 if name == "Zoon0PredV" else 0.65
line_style = lambda name: "-" if name == "Zoon0PredV" else "--"

def ROC_curve(main, ROCS):
    # sns.set_theme(context="paper", style="darkgrid", palette="tab10")
    fig, ax = plt.subplots(figsize=(4,2), dpi=600)


    for _, row in main.iterrows():
        fpr = ROCS[f"FPR_{row['Name']}"]
        tpr = ROCS[f"TPR_{row['Name']}"]
        ax.plot(
            fpr,
            tpr,
            lw=line_width(row["Name"]),
            linestyle=line_style(row["Name"]),
            label="%s (area = %0.2f)" % (row["Name"], row["ROC AUC"])
        )

    ax.plot([0, 1], [0, 1], lw=0.5, linestyle=":")

    ax.set_xlim([0.0, 1.01])
    ax.set_ylim([0.0, 1.05])

    # ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    # ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    ax.set_xticklabels([round(label, 1) for label in ax.get_xticks()], fontdict={"fontsize":4})
    ax.set_yticklabels([round(label, 1) for label in ax.get_yticks()], fontdict={"fontsize":4})

    ax.set_xlabel("False Positive Rate", fontsize=5)
    ax.set_ylabel("True Positive Rate", fontsize=5)

    ax.legend(loc="lower right", fontsize=3)

    plt.tight_layout()

    plt.savefig("Receiver_operating_characteristic.png", bbox_inches='tight', pad_inches=0.0)

def PR_curve(main, PRS):
    fig, ax = plt.subplots(figsize=(4,2), dpi=600)

    for _, row in main.iterrows():
        precision =  PRS[f"Precision_{row['Name']}"]
        recall = PRS[f"Recall_{row['Name']}"]

        ax.plot(
            precision,
            recall,
            lw=line_width(row["Name"]),
            linestyle=line_style(row["Name"]),
            label="%s (F1 score = %0.2f)" % (row["Name"], row["F1 Score"])
        )

    ax.plot([0.5, 0], [1, 1], lw=0.5, linestyle=":")

    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.6, 1.01])

    # ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    # ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    ax.set_xlabel("Precision", fontsize = 5)
    ax.set_ylabel("Recall", fontsize = 5)

    ax.set_yticklabels([round(label, 1) for label in ax.get_yticks()], fontdict={"fontsize":4})
    ax.set_xticklabels([round(label, 1) for label in ax.get_xticks()], fontdict={"fontsize":4})

    ax.legend(loc="lower left", fontsize=3)

    plt.tight_layout()


    plt.savefig("Precision_recall_curve.png", bbox_inches='tight', pad_inches=0.0)


#################### Programm starts here ##################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Zoonosis Model Evaluation',
                                     description=\
                                     """
                                     Test the trained neural network zoonosis prediction model
                                     performance
                                     """.strip(),
                                     )

    ## Could allow list input but nah!!
    parser.add_argument('-d', '--directory',
                        required=True,
                        help='Main directory containing zoonosis data subsplit directories')

    args = parser.parse_args()

    base_dir = args.directory

    main_dir = base_dir

    test_dirs = os.listdir(main_dir)

    test_dirs = [os.path.join(main_dir, fol) for fol in test_dirs]

    test_dirs = list(filter(lambda x: os.path.isdir(x), test_dirs))

    models = [os.path.join(path, 'model') for path in test_dirs]
    test_data = [os.path.join(path, 'test') for path in test_dirs]

    main = pd.DataFrame()
    PRS = dict()
    ROCS = dict()

    for model_, test_data in zip(models, test_data):
        model = load_compile_model(model_)
        metrics, PR, ROC = test_model(model, test_data)
        main = pd.concat([main, metrics], sort=False)
        PRS.update(PR)
        print(model_, end="\n")
        print("\n")
        ROCS.update(ROC)



    main.to_csv("Model_performance.csv", index=False)

    sns.set_theme(context="paper", style="darkgrid", palette="tab10")


    PR_curve(main, PRS) # Saves precision-recall plot in png figure
    print("Done with ROC curve")

    ROC_curve(main, ROCS) # Saves ROC plot in png figure
    print("Done with ROC curve")