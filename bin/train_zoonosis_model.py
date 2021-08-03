#!/opt/conda/envs/tensorEnv/bin/python
import argparse
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint#, EarlyStopping
# from tensorflow.keras.metrics import Recall, Precision
# tf.get_logger().setLevel('ERROR')
# Set GPU usage and memory growth
physical_devices = tf.config.experimental.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, enable=True) 

mirrored_strategy = tf.distribute.MirroredStrategy() # For multi gpu training
# distributed_strategy = tf.distribute.Strategy(extended)

def loadImages(train):
    """
    params:
        train      : directory name for training data
    returns:
        A tuple of training and validation image data generator objects of 128 batches 
    """
    data_generator = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_data_iterator = data_generator.flow_from_directory(
        train,
        target_size=(96,96),
        color_mode='grayscale',
        classes=['human-false', 'human-true'],
        class_mode='binary',
        batch_size=128,
        seed=19980603,
        subset = 'training'
    )
    
    validation_data_iterator = data_generator.flow_from_directory(
        train,
        target_size=(96,96),
        color_mode='grayscale',
        classes=['human-false', 'human-true'],
        class_mode='binary',
        batch_size=128,
        seed=19980603,
        subset = 'validation'
    )

    return (train_data_iterator, validation_data_iterator)


def trainSaveModel(base_dir, train_data_iterator, validation_data_iterator, model_checkpoint, logs):
    """
    params:
        base_dir                    : Base directory containing training, test and validation datasets and directory for saving
        train_data_iterator         : ImageData generator object for the training data, produced by the loadImages function
        validation_data_iterator    : ImageData generator object for the validation data, produced by the loadImages function
        model_checkpoint            : Name to give the model at checkpoints
        logs                        : Name to save the csv training logs for each epoch
    returns:
        saves best model in directory after training alongside training logs in csv
    """
#     steps_per_epoch = train_data_iterator
#     validation_steps = validation_data_iterator
    model_checkpoint_callback = ModelCheckpoint(
        filepath=model_checkpoint,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True
    )
    # stop_early_callback = EarlyStopping(
    #     monitor='val_accuracy',
    #     patience=5
    # )
    TRAIN_STEPS = (train_data_iterator.samples // 128)
    VALIDATION_STEPS = (validation_data_iterator.samples // 128)

    with mirrored_strategy.scope():
        model = keras.models.Sequential([
            keras.layers.Conv2D(48, input_shape=(96,96,1),
                                kernel_size=(3,3),
                                activation='relu',
                                bias_regularizer=tf.keras.regularizers.l2()),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(48, kernel_size=(3,3), activation='relu',
                                bias_regularizer=tf.keras.regularizers.l2()),
            keras.layers.MaxPool2D(pool_size=(2,2), strides=2),
            keras.layers.Dropout(0.2),
            keras.layers.Flatten(),
            keras.layers.Dense(units=128, activation='relu',
                            bias_regularizer=tf.keras.regularizers.l2()),
            keras.layers.Dense(1, activation='sigmoid',
                            bias_regularizer=tf.keras.regularizers.l2())
        ])
        # Precision and recall should be defined under the same scope as the model
        # precision = Precision()
        # recall = Recall()

    model.compile(
        loss='binary_crossentropy',
        optimizer='rmsprop',
        metrics=[
            'accuracy',
            # precision,
            # recall
            ])

    model.fit(
        train_data_iterator, 
        steps_per_epoch=TRAIN_STEPS,
        validation_data=validation_data_iterator, 
        validation_steps=VALIDATION_STEPS,
        shuffle=True,
        epochs=30, 
        verbose=0, 
        callbacks=[
            CSVLogger(logs),
            model_checkpoint_callback,
            # stop_early_callback
            ])
    
    model.load_weights(model_checkpoint)
    keras.models.save_model(model, os.path.join(base_dir, 'model'))


def main(base_dir, train, model_checkpoint, logs):
    
    train_iterator, validation_data_iterator = loadImages(train)
    
    trainSaveModel(base_dir, train_iterator, validation_data_iterator, model_checkpoint, logs)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Zoonosis Model Training',
                                     description=\
                                     """
                                     Train neural network for zoonosis prediction using CGR images
                                     """.strip(),
                                     )
    
    parser.add_argument('-d', '--baseDirectory',
                        required=True,
                        help='Directory containing CGR images split into train, test and validation sub-directories')
        
    parser.add_argument('-t', '--train', default=None,
                        required=False,
                        help='Directory containing traininig CGR images')
            
    parser.add_argument('-m', '--model_checkpoint', default=None,
                        required=False,
                        help='Directory to save best model, default saves to base directory')
            
    parser.add_argument('-l', '--logs', default=None,
                        required=False,
                        help='Directory to save training logs as csv, default saves to base directory')

    args = parser.parse_args()

    # assert os.path.isdir(args.baseDirectory), 'This directory does not exist'
    base_dir = args.baseDirectory
    
    if (args.train == None):
        train = os.path.join(base_dir, 'train')  
    else:
        assert os.path.isdir(args.train), 'This directory does not exist'
        train = args.train

    if args.model_checkpoint == None:
        model_checkpoint = os.path.join(base_dir, 'checkpoint.ckpt')
    else:
        model_checkpoint = args.model
    
    if (args.logs == None):
        logs = os.path.join(base_dir, 'trainingLogs.csv')
    else:
        logs = args.logs
        
    main(base_dir, train, model_checkpoint, logs)