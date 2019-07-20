#!/usr/bin/env python
# coding: utf-8

__DEBUG__ = True


import io
import itertools
import os
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"]="-1" #set to -1 to disable gpu (on fresh kernel)

import tensorflow as tf
import keras
import keras.backend as K

# config = tf.ConfigProto(inter_op_parallelism_threads=6)

# session = tf.Session(config=config)

# K.set_session(session)

# os.environ["OMP_NUM_THREADS"] = "NUM_PARALLEL_EXEC_UNITS"
# os.environ["OMP_NUM_THREADS"] = '24'
#
# os.environ["KMP_BLOCKTIME"] = "30"
#
# os.environ["KMP_SETTINGS"] = "1"
#
# os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Dense, LSTM
from keras.optimizers import Adam
from keras import regularizers

from keras.layers import Input, Dropout
from keras.models import Model


from keras.callbacks import EarlyStopping
from keras.layers import TimeDistributed, Bidirectional

import json
import sys

from utility.TrainValTensorBoardCallBack import TrainValTensorBoard
from keras.utils import plot_model

#constants (for now)
CAMERA_FEATURES = 58
BATCH_SIZE = 1


def get_prepare_data(df, TIMESTEPS):
    def prepare_data(mode="train"):

        if mode == "train":
            while True:
                for i in range(0, L - TIMESTEPS):
                    yield np.reshape(df.iloc[i:i + TIMESTEPS, :].values, (1, TIMESTEPS, CAMERA_FEATURES)), np.reshape(
                        df.iloc[i + TIMESTEPS, :].values, (1, CAMERA_FEATURES))
        elif mode == "eval":
            while True:
                for i in range(L - TIMESTEPS, len(df) - TIMESTEPS):
                    yield np.reshape(df.iloc[i:i + TIMESTEPS, :].values, (1, TIMESTEPS, CAMERA_FEATURES)), np.reshape(
                        df.iloc[i + TIMESTEPS, :].values, (1, CAMERA_FEATURES))
        elif mode == "predict":
            while True:
                for i in range(L - TIMESTEPS, len(df) - TIMESTEPS):
                    yield np.reshape(df.iloc[i:i + TIMESTEPS, :].values, (1, TIMESTEPS, CAMERA_FEATURES))

    return prepare_data


def get_lstm_layers(params):
    """returns keras tensor output of a block of stateful lstms (potentially stacked)."""
    n = params["stack_lstm_count"]

    layers = []

    lstm1 = LSTM(params["lstm1_units"],
                 batch_input_shape=(1,
                                    params["timesteps"],
                                    params["td{}_dense_units".format(params["depth_timedist_dense"])]),
                 stateful=True,
                 return_sequences=True if n > 1 else False,
                 dropout=params["lstm1_dropout"],
                 recurrent_dropout=params["lstm1_recurrent_dropout"],
                 activity_regularizer=regularizers.l2(l=params["lstm1_l2reg"]),
                 )

    if "bidirectional_lstms" in params and params["bidirectional_lstms"] == True:
        lstm1 = Bidirectional(lstm1)  # default mode is concat

    layers.append(lstm1)

    if n == 1:
        return layers

    lstm2 = LSTM(params["lstm2_units"],
                 stateful=True,
                 return_sequences=True if n > 2 else False,
                 dropout=params["lstm2_dropout"],
                 recurrent_dropout=params["lstm2_recurrent_dropout"],
                 activity_regularizer=regularizers.l2(l=params["lstm2_l2reg"]),
                 )

    if "bidirectional_lstms" in params and params["bidirectional_lstms"] == True:
        lstm2 = Bidirectional(lstm2)  # default mode is concat

    layers.append(lstm2)

    if n == 2:
        return layers

    lstm3 = LSTM(params["lstm3_units"],
                 stateful=True,
                 return_sequences=False,
                 dropout=params["lstm3_dropout"],
                 recurrent_dropout=params["lstm3_recurrent_dropout"],
                 activity_regularizer=regularizers.l2(l=params["lstm3_l2reg"]),
                 )

    if "bidirectional_lstms" in params and params["bidirectional_lstms"] == True:
        lstm3 = Bidirectional(lstm3)  # default mode is concat

    layers.append(lstm3)

    return layers


def get_time_dist_dense_layers(params):
    layers = []
    for i in range(1, params["depth_timedist_dense"]+1 if "depth_timedist_dense" in params else 2):
        layers.append(TimeDistributed(Dense(params["td{}_dense_units".format(i)],
                                      activation=params["td{}_dense_activation".format(i)],
                                      activity_regularizer=regularizers.l2(l=params["td{}_dense_l2reg".format(i)]))))
        layers.append(TimeDistributed(Dropout(rate=params["td{}_dense_dropout".format(i)])))

    return layers


def get_projection_dense_layers(params):
    layers = []
    for i in range(1, params["depth_projection_dense"]+1 if "depth_projection_dense" in params else 2):
        layers.append( Dense(params["dense{}_units".format(i)],
                             activation=params["dense{}_activation".format(i)],
                             activity_regularizer=regularizers.l2(l=params["dense{}_l2reg".format(i)])) )
        layers.append(Dropout(rate=params["dense{}_dropout".format(i)]))
    return layers


def get_model(params):
    matrix_in = Input(batch_shape=(1, params["timesteps"], CAMERA_FEATURES), )


    concat = matrix_in
    for tdlayer in get_time_dist_dense_layers(params):
        concat = tdlayer(concat)


    lstm_out = concat
    for lstm in get_lstm_layers(params):
        lstm_out = lstm(lstm_out)


    out = lstm_out
    for dlayer in get_projection_dense_layers(params):
        out = dlayer(out)


    #last one project to number of cameras against loss.
    out = Dense(CAMERA_FEATURES, activation=params["out_dense_activation"],
                 activity_regularizer=regularizers.l2(l=params["out_dense_l2reg"])
                 )(out)


    return Model(inputs=matrix_in, outputs=out)


def exp_loss(yTrue,yPred):
    return K.mean(K.exp(K.square(yTrue - yPred,)))


def _plot_y_vs_pred_on_BytesIO(y,predictions):
    plt.figure()
    plt.plot(y)
    plt.plot(predictions)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_string = buf.getvalue()
    buf.close()
    plt.close()
    return tf.Summary.Image(encoded_image_string = image_string)


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("usage: python train_lstm_for_london.py paramss.json")
        print("paramss.json must be compatible with list of dictionaries.")
        sys.exit()

    paramss_path = os.path.abspath(sys.argv[1]) #absolute so we can reuse it later even if we change working dir.
    with open(paramss_path, 'r') as f:
        paramss = json.load(f)

    print("number of configs to try:", len(paramss))

    print("checking gpu:")
    print(keras.backend.tensorflow_backend._get_available_gpus())
    # tf.test.is_gpu_available()

    os.chdir(r"../immediate_results")

    df = pd.read_csv(r"cleandf_40days_930_to_1830_lin_interpolated.csv")

    # df.DENSITY_VALUE = np.log(df.DENSITY_VALUE)
    df.DENSITY_VALUE -= df.DENSITY_VALUE.min()
    df.DENSITY_VALUE /= df.DENSITY_VALUE.max()
    df = df.pivot(index="TIMESTAMP",columns="CAMERA_ID",values="DENSITY_VALUE")
    L = int(0.75 * len(df))


    CURRENT_TIME = datetime.now().strftime("%Y-%m-%d %H-%M")

    for params in paramss:


        if __DEBUG__:
            print(params)

        if "_ignore" in params and params["_ignore"] == True:
            print("_ignore flag set in config. Skipping this config.")
            continue

        model = get_model(params)

        model.compile(optimizer=Adam(lr=0.001,clipnorm=1.),
                      loss=exp_loss if params["loss"] == "exp_loss" else params["loss"])
                      # metrics=["mse"])

        print(model.summary())
        if "plot_model?" in params and params["plot_model?"] == True:
            plot_model(model, to_file=params["model_name"] + ".png", show_shapes=True)

        logdir = "logs/{}[{}]".format(params["desc"], CURRENT_TIME )

        prepare_data = get_prepare_data(df, params["timesteps"]) #partial function

        # tensorboard = TrainValTensorBoard(log_dir= logdir)
        #, write_grads=True, histogram_freq= 2, batch_size=1 )

        es = EarlyStopping(monitor="val_loss", patience=10,
                           min_delta=params["min_delta"] if "min_delta" in params else 0.0001,
                           restore_best_weights=True)

        history = model.fit_generator(prepare_data(mode="train"),
                                      epochs = 200,
                                      steps_per_epoch= L-params["timesteps"],
                                      validation_data = prepare_data(mode="eval"),
                                      validation_steps = len(df) - L,
                                      callbacks=[es],
                                      verbose=2
                                     )

        one_test = itertools.islice(prepare_data(mode="eval"), len(df) - L)
        a = list(one_test)
        X, y = zip(*a)
        y = np.array(y).reshape((len(df) - L, CAMERA_FEATURES)) #original y was (df-l,1,cam_feat)
        X = np.array(X).reshape((len(df) - L, -1, CAMERA_FEATURES))


        # val_writer_path = os.path.join(logdir, 'validation')
        # val_writer = tf.summary.FileWriter(val_writer_path)
        predictions = model.predict(X,batch_size=1)

        np.save("eval_X_nni_best_multivariate.npy", X)
        np.save("eval_y_nni_best_multivariate.npy", y)
        np.save("eval_pred_nni_best_multivariate.npy", predictions)

        mses = []
        for i in range(y.shape[1]):
            mses.append(np.mean(np.square(y[:,i] - predictions[:,i])))

        print(np.mean(mses))

        # with open("lstm_all_city_results.txt",'a') as f:
        #     f.writelines(["{} {}\n".format(i,mse) for i, mse in enumerate(mses)])

        # for i in [0, 1, 10, 20]: #camera indices
        #
        #     img = _plot_y_vs_pred_on_BytesIO(y[:, i], predictions[:, i])
        #     summary_pb = tf.Summary(value=[tf.Summary.Value(tag="camera_{}_prediction_vs_actual".format(i), image=img)])
        #     val_writer.add_summary(summary_pb)
        #
        #
        #
        # text_tensor = tf.make_tensor_proto([(k,str(v)) for k,v in params.items()], dtype=tf.string)
        # meta = tf.SummaryMetadata()
        # meta.plugin_data.plugin_name = "text"
        # summary_pb = tf.Summary()
        # summary_pb.value.add(tag="parameters", metadata=meta, tensor=text_tensor)
        # val_writer.add_summary(summary_pb)
        # val_writer.close()

        params["_ignore"] = True


        if "save?" in params and params["save?"] == True:
            model.save("{}[{}].h5".format(params["desc"], CURRENT_TIME))

        del model
        del history
        del prepare_data
        K.clear_session() #to hopefully prevent slow down after a few models..

    #writing the log file back with potential info about the run (including _ignore flag which is set after it is run
    # once).
    with open(paramss_path,'w') as f:
        json.dump(paramss,f)