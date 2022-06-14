#!/usr/local/bin/python

import argparse
import os
import re
import tensorflow as tf
# from tensorflow import keras
from tensorflow.random import set_seed
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau #, EarlyStopping
# from tensorflow.keras.metrics import  BinaryAccuracy, FalsePositives, TruePositives, TrueNegatives, \
#                                       FalseNegatives, AUC , Recall, Precision
import numpy as np # Needed by reduce_lr
from numpy.random import seed
from model_definition import Zoon0Pred_model
# tf.get_logger().setLevel('ERROR')

# Set random state for tensorflow operations (REPRODUCIBILITY)
set_seed(20192020)
seed(20192020)

# Set GPU usage and enable memory growth
physical_devices = tf.config.experimental.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, enable=True)

mirrored_strategy = tf.distribute.MirroredStrategy() # For multi gpu training
# distributed_strategy = tf.distribute.Strategy(extended)

def loadImages(train, BATCH_SIZE, TARGET_SIZE=(224,224)):
    """
    params:
        train      : directory name for training data
    returns:
        A tuple of training and validation image data generator objects in batches of 256
    """
    data_generator = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_data_iterator = data_generator.flow_from_directory(
        train,
        target_size=TARGET_SIZE,
        color_mode='grayscale',
        classes=['human-false', 'human-true'],
        class_mode='binary',
        batch_size=BATCH_SIZE,
        seed=19980603,
        subset = 'training'
    )

    validation_data_iterator = data_generator.flow_from_directory(
        train,
        target_size=TARGET_SIZE,
        color_mode='grayscale',
        classes=['human-false', 'human-true'],
        class_mode='binary',
        batch_size=BATCH_SIZE,
        seed=19980603,
        subset = 'validation'
    )

    return (train_data_iterator, validation_data_iterator)


def trainSaveModel(train_data_iterator, validation_data_iterator, model_checkpoint, logs, model_name, BATCH_SIZE):
    """
    params:
        train_data_iterator         : ImageData generator object for the training data, produced by the loadImages function
        validation_data_iterator    : ImageData generator object for the validation data, produced by the loadImages function
        model_checkpoint            : Name to give the model at checkpoints
        logs                        : Name to save the csv training logs for each epoch
    returns:
        saves best model in directory after training alongside training logs in csv
    """

    model_checkpoint_callback = ModelCheckpoint(
        filepath=model_checkpoint,
        save_weights_only=True,
        monitor='val_MCC',
        mode='max',
        save_best_only=True
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_AUC',
        factor=0.002, patience=3, # Reduce learning rate by 0.002 if the AUC remains unchanged for 3 epochs
        min_lr=0.001
    )

    # stop_early_callback = EarlyStopping(
    #     monitor='val_accuracy',
    #     patience=5
    # )
    TRAIN_STEPS = int(round(train_data_iterator.samples / BATCH_SIZE))
    VALIDATION_STEPS = int(round(validation_data_iterator.samples / BATCH_SIZE))

    model = Zoon0Pred_model(mirrored_strategy)

    model.fit(
        train_data_iterator,
        steps_per_epoch=TRAIN_STEPS,
        validation_data=validation_data_iterator,
        validation_steps=VALIDATION_STEPS,
        shuffle=True,
        epochs=50,
        verbose=0,
        use_multiprocessing=True,
        callbacks=[
            CSVLogger(logs),
            model_checkpoint_callback,
            reduce_lr
            # stop_early_callback
            ]
    )
    model.load_weights(model_checkpoint)
    tf.keras.models.save_model(model, model_name)


def main(train, model_checkpoint, logs, model_name, BATCH_SIZE=64):

    train_iterator, validation_data_iterator = loadImages(train, BATCH_SIZE)

    trainSaveModel(train_iterator, validation_data_iterator, model_checkpoint, logs, model_name, BATCH_SIZE)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Zoonosis Model Training',
                                     description=\
                                     """
                                     Train neural network for zoonosis prediction using CGR images
                                     """.strip(),
                                     )

    parser.add_argument('-d', '--baseDirectory',
                        required=False,
                        help='Main directory containing train and test sub-directories')

    parser.add_argument('-t', '--train', default=None,
                        required=False,
                        help='Directory containing traininig CGR images')

    parser.add_argument('-m', '--model_checkpoint', default=None,
                        required=False,
                        help='Name of the model, default saves to base directory')

    parser.add_argument('-l', '--logs', default=None,
                        required=False,
                        help='Training logs as csv, default saves to base directory')

    parser.add_argument('-n', '--name', default=None,
                        required=False,
                        help='Name to save model. Default: model')

    parser.add_argument('-b', '--batch_size', default=None,
                        required=False, type=int,
                        help='Name to save model. Default: model')

    args = parser.parse_args()

    base_dir = args.baseDirectory

    if (args.train == None):
        train = os.path.join(base_dir, 'train')
    else:
        assert os.path.isdir(args.train), 'This directory does not exist'
        train = args.train

    if (args.model_checkpoint == None):
        model_checkpoint = 'checkpoint.ckpt'
    else:
        model_checkpoint = args.model

    if (args.logs == None) & (base_dir != None):
        logs = f'{base_dir}_trainingLogs.csv'
    else:
        logs = args.logs

    if (args.name == None) & (base_dir != None):
        model_name = os.path.join(base_dir, 'model')
    else:
        model_name = args.name

    # # Used for testing... different batch size testing
    # if (args.batch_size == None):
    #     BATCH_SIZE = 64
    # else:
    #     BATCH_SIZE = args.batch_size


    # directory = base_dir.split("/")[-1]
    # condition =  or directory == "RNA-MetazoaZoonosisData"
    # if (base_dir == "Zoon0PredV"):
    #     BATCH_SIZE=128
    # elif (base_dir == "Metazoa") | (base_dir.__contains__("RNA")):
    #     BATCH_SIZE=64
    # else:
    #     BATCH_SIZE=32
    BATCH_SIZE=64
    main(train, model_checkpoint, logs, model_name, BATCH_SIZE=BATCH_SIZE)
