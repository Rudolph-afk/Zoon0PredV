
�2root"_tf_keras_sequential*�2{"name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 224, 224, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 224, 224, 1]}, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "shared_object_id": 11, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}, "shared_object_id": 12}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 224, 224, 1]}, "is_graph_network": true, "full_save_spec": {"class_name": "__tuple__", "items": [[{"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 224, 224, 1]}, "float32", "conv2d_input"]}], {}]}, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 224, 224, 1]}, "float32", "conv2d_input"]}, "keras_version": "2.6.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 224, 224, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_input"}, "shared_object_id": 0}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 224, 224, 1]}, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 3}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 4}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 5}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 6}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 9}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 10}]}}, "training_config": {"loss": "binary_crossentropy", "metrics": [[{"class_name": "BinaryAccuracy", "config": {"name": "binary_accuracy", "dtype": "float32", "threshold": 0.5}, "shared_object_id": 13}, {"class_name": "FalsePositives", "config": {"name": "FP", "dtype": "float32", "thresholds": 0.5}, "shared_object_id": 14}, {"class_name": "TruePositives", "config": {"name": "TP", "dtype": "float32", "thresholds": 0.5}, "shared_object_id": 15}, {"class_name": "TrueNegatives", "config": {"name": "TN", "dtype": "float32", "thresholds": 0.5}, "shared_object_id": 16}, {"class_name": "FalseNegatives", "config": {"name": "FN", "dtype": "float32", "thresholds": 0.5}, "shared_object_id": 17}, {"class_name": "MatthewsCorrelationCoefficient", "config": {"name": "MCC", "dtype": "float32", "thresholds": 0.5, "top_k": null, "class_id": null}, "shared_object_id": 18}, {"class_name": "AUC", "config": {"name": "AUC", "dtype": "float32", "num_thresholds": 200, "curve": "ROC", "summation_method": "interpolation", "multi_label": false, "label_weights": null}, "shared_object_id": 19}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}2
�root.layer_with_weights-0"_tf_keras_layer*�
{"name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 224, 224, 1]}, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 224, 224, 1]}, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 3}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 4, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}, "shared_object_id": 12}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 224, 224, 1]}}2
�root.layer-1"_tf_keras_layer*�{"name": "max_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 5, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 20}}2
�root.layer-2"_tf_keras_layer*�{"name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 21}}2
�root.layer_with_weights-1"_tf_keras_layer*�{"name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 9}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 10, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1577088}}, "shared_object_id": 22}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1577088]}}2
�=root.keras_api.metrics.0"_tf_keras_metric*�{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 23}2
�>root.keras_api.metrics.1"_tf_keras_metric*�{"class_name": "BinaryAccuracy", "name": "binary_accuracy", "dtype": "float32", "config": {"name": "binary_accuracy", "dtype": "float32", "threshold": 0.5}, "shared_object_id": 13}2
�?root.keras_api.metrics.2"_tf_keras_metric*�{"class_name": "FalsePositives", "name": "FP", "dtype": "float32", "config": {"name": "FP", "dtype": "float32", "thresholds": 0.5}, "shared_object_id": 14}2
�@root.keras_api.metrics.3"_tf_keras_metric*�{"class_name": "TruePositives", "name": "TP", "dtype": "float32", "config": {"name": "TP", "dtype": "float32", "thresholds": 0.5}, "shared_object_id": 15}2
�Aroot.keras_api.metrics.4"_tf_keras_metric*�{"class_name": "TrueNegatives", "name": "TN", "dtype": "float32", "config": {"name": "TN", "dtype": "float32", "thresholds": 0.5}, "shared_object_id": 16}2
�Broot.keras_api.metrics.5"_tf_keras_metric*�{"class_name": "FalseNegatives", "name": "FN", "dtype": "float32", "config": {"name": "FN", "dtype": "float32", "thresholds": 0.5}, "shared_object_id": 17}2
�Croot.keras_api.metrics.6"_tf_keras_metric*�{"class_name": "MatthewsCorrelationCoefficient", "name": "MCC", "dtype": "float32", "config": {"name": "MCC", "dtype": "float32", "thresholds": 0.5, "top_k": null, "class_id": null}, "shared_object_id": 18}2
�Droot.keras_api.metrics.7"_tf_keras_metric*�{"class_name": "AUC", "name": "AUC", "dtype": "float32", "config": {"name": "AUC", "dtype": "float32", "num_thresholds": 200, "curve": "ROC", "summation_method": "interpolation", "multi_label": false, "label_weights": null}, "shared_object_id": 19}2