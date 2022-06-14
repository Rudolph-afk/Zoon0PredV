import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.metrics import  BinaryAccuracy, FalsePositives, TruePositives, TrueNegatives, \
                                      FalseNegatives, AUC , Recall, Precision
from metrics_helper import MatthewsCorrelationCoefficient

# mirrored_strategy = tf.distribute.MirroredStrategy() # For multi gpu training

def Zoon0Pred_model(mirrored_strategy, thresh=0.5, cnn_units=128, cnn_kernel_size=(2,2), max_pool_size=(2,2)):

    with mirrored_strategy.scope():
        model = keras.models.Sequential([
            keras.layers.Conv2D(
                cnn_units, input_shape=(224,224,1), kernel_size=cnn_kernel_size, activation='relu',
                bias_regularizer=tf.keras.regularizers.l2()),
            keras.layers.MaxPool2D(pool_size=max_pool_size,),

            # keras.layers.BatchNormalization(),

            # keras.layers.Conv2D(
            #     cnn_units*2, kernel_size=cnn_kernel_size, activation='relu',
            #     bias_regularizer=tf.keras.regularizers.l2()),
            # keras.layers.MaxPool2D(pool_size=max_pool_size),

            # keras.layers.BatchNormalization(),

            # keras.layers.Conv2D(
            #     cnn_units*3, kernel_size=cnn_kernel_size, activation='relu',
            #     bias_regularizer=tf.keras.regularizers.l2()),
            # keras.layers.MaxPool2D(pool_size=max_pool_size),

            keras.layers.Flatten(),

            keras.layers.Dense(
                1, activation='sigmoid',
                bias_regularizer=tf.keras.regularizers.l2())
            ]
        )
        accuracy = BinaryAccuracy(threshold=thresh)
        # recall = Recall(name = "Recall", thresholds=thresh)
        # precision = Precision(name = "Precision", thresholds=thresh)
        false_positives = FalsePositives(name="FP",thresholds=thresh)
        true_positives = TruePositives(name="TP", thresholds=thresh)
        true_negatives = TrueNegatives(name="TN", thresholds=thresh)
        false_negatives = FalseNegatives(name="FN", thresholds=thresh)
        methewsCC = MatthewsCorrelationCoefficient(name="MCC", thresholds=thresh)
        auc_roc = AUC(name='AUC')

    model.compile(
        loss='binary_crossentropy',
        optimizer=keras.optimizers.Adam(learning_rate=0.019),
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

def load_compile_model(model_dir: str, thresh=0.5):
    model = tf.keras.models.load_model(model_dir, compile=False)

    accuracy = BinaryAccuracy(threshold=thresh)
    false_positives = FalsePositives(name="FP", thresholds=thresh)
    true_positives = TruePositives(name="TP", thresholds=thresh)
    true_negatives = TrueNegatives(name="TN", thresholds=thresh)
    false_negatives = FalseNegatives(name="FN", thresholds=thresh)
    methewsCC = MatthewsCorrelationCoefficient(name="MCC", thresholds=thresh)
    auc_roc = AUC(name='AUC')
    recall = Recall(name = "Recall", thresholds=thresh)
    precision = Precision(name = "Precision", thresholds=thresh)

    model.compile(
        loss='binary_crossentropy',
        optimizer=keras.optimizers.Adam(learning_rate=0.019),
        metrics=[
            accuracy,
            false_positives,
            true_positives,
            true_negatives,
            false_negatives,
            methewsCC,
            auc_roc,
            recall,
            precision
        ]
    )
    return model
