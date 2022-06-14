#!/usr/local/bin/python

import argparse
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
from tensorflow.keras.callbacks import CSVLogger, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.metrics import Recall, Precision, BinaryAccuracy, FalsePositives, \
                                    TruePositives, TrueNegatives, FalseNegatives, AUC
from metrics_helper import MatthewsCorrelationCoefficient
from tensorflow.random import set_seed
from numpy.random import seed
from sklearn.model_selection import train_test_split
import random
from tqdm.notebook import tqdm
import pandas as pd
from tensorflow.keras.utils import plot_model

random.seed(20192020)
# set_seed(20192020)
# seed(20192020)

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print('Number of GPUs available: ', len(physical_devices))
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, enable=True)

# mirrored_strategy = tf.distribute.MirroredStrategy()

def createImageData(data_dir, subdir, img_size=224):
    data = []
    Features = []
    Labels = []
    with tqdm(total=len(subdir), desc="Directory progress", position=0) as pbar:
        for directory in subdir:
            path = os.path.join(data_dir, directory)
            class_num = subdir.index(directory)
            images = list(filter(lambda x: x.endswith('png'), os.listdir(path)))
            random.shuffle(images)
            l=int(len(images)/7)
            images = images[:l]
            with tqdm(total=len(images), desc="Reading images", position=1) as pbar2:
                for image in images:
                    image = os.path.join(path, image)
                    image_as_array = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
                    image_as_array = image_as_array/255.0
                    data.append([image_as_array, class_num])
                    pbar2.update()
            pbar.update()
#     random.shuffle(data)
    with tqdm(total=len(data), desc='Appending features and labels') as pbar3:
        for feature, label in data:
            Features.append(feature)
            Labels.append(label)
            pbar3.update()
    Features = np.array(Features).reshape((-1, img_size, img_size, 1))
#     Labels = np.asarray(Labels).astype('float32').reshape((-1,1))
    Labels = np.array(Labels).reshape((-1,1))
    return (Features, Labels)


def add_con2D(model, HP_NUM_UNITS, HP_NUM_POOL, HP_NUM_KERNEL):
    model.add(
        keras.layers.Conv2D(HP_NUM_UNITS,
                            kernel_size=(HP_NUM_KERNEL,HP_NUM_KERNEL),
                            activation='relu',
                            bias_regularizer=tf.keras.regularizers.l2())
    )
    model.add(
        keras.layers.MaxPool2D(pool_size=(HP_NUM_POOL,HP_NUM_POOL))
    )

def add_last(model):
    model.add(
        keras.layers.Flatten()
    )
    model.add(
            keras.layers.Dense(
                1, activation='sigmoid',
                bias_regularizer=tf.keras.regularizers.l2())
    )

def build_model(hp):
    HP_NUM_UNITS = hp.Int('units', min_value=48, max_value=128, step=16)
    HP_NUM_KERNEL = hp.Int('kernel', min_value=1, max_value=3, step=1)
    HP_NUM_POOL = hp.Int('pool', min_value=1, max_value=3, step=1)
    HP_OPTIMIZER = hp.Choice('optimizer', ['Adam', 'RMSprop'])
    HP_THRESHOLD = hp.Float('threshold', min_value=0.5, max_value=0.9, step=0.05)
    HP_LEARNING_RATE = hp.Float('learning_rate', min_value=0.001, max_value=0.02, step=0.002)
    HP_NUM_LAYERS = hp.Choice("model_layers", ["1 Conv layer", "2 Conv layers", "3 Conv layers"])

    # with mirrored_strategy.scope():
    model = keras.models.Sequential([
        keras.layers.Conv2D(
            HP_NUM_UNITS,
            input_shape=(224,224,1), kernel_size=(HP_NUM_KERNEL,HP_NUM_KERNEL), activation='relu',
            bias_regularizer=tf.keras.regularizers.l2()
        ),
        keras.layers.MaxPool2D(pool_size=(HP_NUM_POOL, HP_NUM_POOL)),
    ])

    if HP_NUM_LAYERS == "1 Conv layer":
        with hp.conditional_scope("model_layers", ["1 Conv layer"]):
                add_last(model)

    if HP_NUM_LAYERS == "2 Conv layers":
        with hp.conditional_scope("model_layers", ["2 Conv layers"]):
            model.add(keras.layers.BatchNormalization())

            # Second layer after input
            add_con2D(model, HP_NUM_UNITS*2, HP_NUM_POOL, HP_NUM_KERNEL)

            add_last(model)

    if HP_NUM_LAYERS == "3 Conv layers":
        with hp.conditional_scope("model_layers", ["3 Conv layers"]):
            model.add(keras.layers.BatchNormalization())

            add_con2D(model, HP_NUM_UNITS*2, HP_NUM_POOL, HP_NUM_KERNEL)

            model.add(keras.layers.BatchNormalization())

            add_con2D(model, HP_NUM_UNITS*3, HP_NUM_POOL, HP_NUM_KERNEL)

            add_last(model)

    thresh = HP_THRESHOLD

    accuracy = BinaryAccuracy(threshold=thresh)
    false_positives = FalsePositives(name="FP",thresholds=thresh)
    true_positives = TruePositives(name="TP", thresholds=thresh)
    true_negatives = TrueNegatives(name="TN", thresholds=thresh)
    false_negatives = FalseNegatives(name="FN", thresholds=thresh)
    methewsCC = MatthewsCorrelationCoefficient(name="MCC", thresholds=thresh)
    auc_roc = AUC(name='AUC')

    if HP_OPTIMIZER == "Adam":
        with hp.conditional_scope("optimizer", ["Adam"]):
                optimizer = tf.keras.optimizers.Adam(learning_rate=HP_LEARNING_RATE)

    if HP_OPTIMIZER == "RMSprop":
        with hp.conditional_scope("optimizer", ["RMSprop"]):
                optimizer = tf.keras.optimizers.RMSprop(learning_rate=HP_LEARNING_RATE)

        # tf.keras.optimizers.Adam(learning_rate=HP_LEARNING_RATE)
        # tf.keras.optimizers.RMSprop(learning_rate=HP_LEARNING_RATE)

    model.compile(
        loss='binary_crossentropy',
        optimizer=HP_OPTIMIZER,
        metrics=[
            accuracy,
            # recall,
            # precision,
            false_positives,
            true_positives,
            true_negatives,
            false_negatives,
            methewsCC,
            auc_roc
        ]
    )
    return model


class MyTuner(kt.tuners.BayesianOptimization):
    def save_model(self, trial_id, model, step=0):
        fname = os.path.join(self.get_trial_dir(trial_id), "model.pickle")
        with tf.io.gfile.GFile(fname, "wb") as f:
            pickle.dump(model, f)

    def update_trial(self, trial, *args, **kwargs):
        super(MyTuner, self).update_trial(trial, *args, **kwargs)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog='Zoonoses Model Hyperparmeter tuning',
        description=\
        """
        Finding the best hyperparameters for the Zoon0Pred model
        """.strip(),
    )

    parser.add_argument(
        '-d', '--data',
        required=True,
        help=\
        """
        Data directory containing zoonosis train data directories with
        human-true and human-false subdirectories containing FCGR images
        """.strip()
    )
    args = parser.parse_args()

    dataset_directory = args.data
    data_subdirectories = ['human-false', 'human-true'] # maintain order (0,1)

    X, y = createImageData(data_dir=dataset_directory, subdir=data_subdirectories)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1997912, test_size=0.25)

    y_train = np.asarray(y_train).astype('float32').reshape((-1,1))
    y_test = np.asarray(y_test).astype('float32').reshape((-1,1))

    # stop_early_callback = EarlyStopping(monitor='val_binary_accuracy', patience=3)
    # logs = CSVLogger("logs.csv")
    # reduce_lr = ReduceLROnPlateau(monitor='val_binary_accuracy', factor=0.2, patience=3, min_lr=0.001)

    # callbacks = [stop_early_callback]

    # Uses same arguments as the BayesianOptimization Tuner.
    tuner = MyTuner(
        build_model,
        distribution_strategy=tf.distribute.MirroredStrategy(),
        seed=71219,
        objective=kt.Objective("val_MCC", direction="max"),
        project_name='BayOptHparams',
        max_trials=500
    )

    print("Starting Serach")
    tuner.search(X_train, y_train, validation_split=0.2, verbose=0, use_multiprocessing=True)

    df = pd.DataFrame()
    for i in range(500):
        tmp = pd.DataFrame([tuner.get_best_hyperparameters(500)[i].values])
        df = pd.concat([df, tmp], ignore_index=True)

    df.to_csv("hyperparameter_tuning.csv", index=False)

    # best_model = tuner.get_best_models()[0]
    # plot_model(best_model, to_file="Best_Model.png", show_layer_names=False, show_shapes=False)
