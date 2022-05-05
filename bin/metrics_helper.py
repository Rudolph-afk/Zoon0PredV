import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras.metrics import Metric
from tensorflow.python.keras.utils import metrics_utils
from tensorflow.python.keras.utils.generic_utils import to_list
import numpy as np

class MatthewsCorrelationCoefficient(Metric):
    """
    MCC = (TP * TN) - (FP * FN) /
          ((TP + FP) * (TP + FN) * (TN + FP ) * (TN + FN))^(1/2)
    Args:
        num_classes : Number of unique classes in the dataset.
        name: (Optional) String name of the metric instance.
        dtype: (Optional) Data type of the metric result.
    Usage:
    """
    def __init__(self,thresholds=None, top_k=None, class_id=None, name=None, dtype=None):
        super(MatthewsCorrelationCoefficient, self).__init__(name=name, dtype=dtype)
        self.init_thresholds = thresholds
        self.top_k = top_k
        self.class_id = class_id

        default_threshold = 0.7 if top_k is None else metrics_utils.NEG_INF
        self.thresholds = metrics_utils.parse_init_thresholds(
            thresholds, default_threshold=default_threshold)
        self._thresholds_distributed_evenly = (
            metrics_utils.is_evenly_distributed_thresholds(self.thresholds))
        self.true_positives = self.add_weight(
            'true_positives',
            shape=(len(self.thresholds),),
            initializer=tf.compat.v1.zeros_initializer)
        self.false_positives = self.add_weight(
            'false_positives',
            shape=(len(self.thresholds),),
            initializer=tf.compat.v1.zeros_initializer)
        self.true_negatives = self.add_weight(
            'true_negatives',
            shape=(len(self.thresholds),),
            initializer=tf.compat.v1.zeros_initializer)
        self.false_negatives = self.add_weight(
            'false_negatives',
            shape=(len(self.thresholds),),
            initializer=tf.compat.v1.zeros_initializer)

    def update_state(self, y_true, y_pred, sample_weight=None):
        return metrics_utils.update_confusion_matrix_variables(
            {
                metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self.true_positives,
                metrics_utils.ConfusionMatrix.FALSE_POSITIVES: self.false_positives,
                metrics_utils.ConfusionMatrix.TRUE_NEGATIVES: self.true_negatives,
                metrics_utils.ConfusionMatrix.FALSE_NEGATIVES: self.false_negatives,
            },
            y_true,
            y_pred,
            thresholds=self.thresholds,
            thresholds_distributed_evenly=self._thresholds_distributed_evenly,
            top_k=self.top_k,
            class_id=self.class_id,
            sample_weight=sample_weight)

    def result(self):
        """
        MCC = (TP * TN) - (FP * FN) /
            ((TP + FP) * (TP + FN) * (TN + FP ) * (TN + FN))^(1/2)
        """
        A = tf.math.subtract(
            tf.math.multiply(self.true_positives, self.true_negatives),
            tf.math.multiply(self.false_positives, self.false_negatives)) # (TP * TN) - (FP * FN)

        B = tf.math.multiply(tf.math.add(self.true_positives, self.false_positives),
                            tf.math.add(self.true_positives, self.false_negatives)) # (TP + FP) * (TP + FN)
        C = tf.math.multiply(tf.math.add(self.true_negatives, self.false_positives),
                            tf.math.add(self.true_negatives, self.false_negatives)) # (TN + FP ) * (TN + FN)

        D = tf.math.sqrt(tf.math.multiply(B, C))


        result = tf.math.divide_no_nan(A, D)

        return result[0] if len(self.thresholds) == 1 else result

    def reset_state(self):
        num_thresholds = len(to_list(self.thresholds))
        backend.batch_set_value([(v, np.zeros((num_thresholds,)))
                                 for v in (self.true_positives,
                                           self.false_positives,
                                           self.true_negatives,
                                           self.false_negatives)])

    def get_config(self):
        config = {
            'thresholds': self.init_thresholds,
            'top_k': self.top_k,
            'class_id': self.class_id
        }
        base_config = super(MatthewsCorrelationCoefficient, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
