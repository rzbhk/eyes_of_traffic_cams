from keras.callbacks import TensorBoard
from os import mkdir, path
import tensorflow as tf
import matplotlib.pyplot as plt
import io
import itertools
import numpy as np


# def _plot_pred_and_actual(model, eval_generator, i):
#     one_test = itertools.islice(eval_generator, len(df) - L)
#     a = list(one_test)
#     X, y = zip(*a)
#     y = np.array(y).reshape((len(df) - L, CAMERA_FEATURES)) #original y was (df-l,1,cam_feat)
#     X = np.array(X).reshape((len(df) - L, -1, CAMERA_FEATURES))
#     predictions = model.predict(X,batch_size=1)
#     return _plot_i(i,y,predictions)


class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir='./logs', **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        mkdir(log_dir)
        # self.generator = generator

        training_log_dir = path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = path.join(log_dir, 'validation')

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)

        # if not self.model:
        #     print("Model not found in custom tensorboard callback.")
        #     self.val_writer.close()
        #     return

        # img = _plot_pred_and_actual(self.model, self.generator, 10)  # 10 is arbitrary (10th camera)
        # summary_op = tf.Summary(value=[tf.Summary.Value(tag="prediction_vs_actual", image=img)])

        # self.val_writer.add_summary(summary_op)
        self.val_writer.close()
