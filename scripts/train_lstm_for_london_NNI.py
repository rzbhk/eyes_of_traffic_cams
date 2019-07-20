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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Dense, LSTM
from keras.optimizers import Adam
from keras import regularizers

from keras.layers import Input, Dropout
from keras.models import Model

import keras.backend as K
from keras.callbacks import EarlyStopping, LambdaCallback

from keras.layers import TimeDistributed, Bidirectional

import json
import sys

# from TrainValTensorBoardCallBack import TrainValTensorBoard

import nni

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


# def exp_loss(yTrue,yPred):
#     return K.mean(K.exp(K.square(yTrue - yPred,)))


# def _plot_y_vs_pred_on_BytesIO(y,predictions):
#     plt.figure()
#     plt.plot(y)
#     plt.plot(predictions)
#     buf = io.BytesIO()
#     plt.savefig(buf, format='png')
#     buf.seek(0)
#     image_string = buf.getvalue()
#     buf.close()
#     plt.close()
#     return tf.Summary.Image(encoded_image_string = image_string)


#this can be done recursively..
def parse_raw_params(raw_params):
    params = {}

    params["depth_timedist_dense"] = 1
    td2 = raw_params["td2"]
    if td2['_name'] != 'Empty':
        params["depth_timedist_dense"] = 2

        for key, val in td2.items():
            if key != '_name':
                params[key] = val

    params['stack_lstm_count'] = 1
    lstm2 = raw_params['lstm2']
    if lstm2['_name'] != 'Empty':
        params['stack_lstm_count'] = 2

        for key, val in lstm2.items():
            if key != '_name':
                params[key] = val

    params['depth_projection_dense'] = 0
    dense1 = raw_params["dense1"]
    if dense1['_name'] != 'Empty':
        params['depth_projection_dense'] = 1

        for key, val in dense1.items():
            if key != '_name' and key != 'dense2':
                params[key] = val

        dense2 = dense1['dense2']
        if dense2['_name'] != 'Empty':
            params['depth_projection_dense'] = 2

            for key, val in dense2.items():
                if key != '_name':
                    params[key] = val

    for key, value in raw_params.items():
        if key == 'td2' or key == 'dense1' or key == 'lstm2':
            continue

        params[key] = value


    for key, value in params.items():
        if "unit" in key:
            params[key] = int(value)

    params["timesteps"] = int(params["timesteps"])

    return params


if __name__ == "__main__":

    os.chdir(r"../immediate_results")

    df = pd.read_csv(r"cleandf_40days_930_to_1830_lin_interpolated.csv")

    # df.DENSITY_VALUE = np.log(df.DENSITY_VALUE)
    df.DENSITY_VALUE -= df.DENSITY_VALUE.min()
    df.DENSITY_VALUE /= df.DENSITY_VALUE.max()
    df = df.pivot(index="TIMESTAMP",columns="CAMERA_ID",values="DENSITY_VALUE")
    L = int(0.75 * len(df))

    raw_params = nni.get_next_parameter()
    print(raw_params)
    params = parse_raw_params(raw_params)

    print(params)
    model = get_model(params)

    model.compile(optimizer=Adam(lr=0.001,clipnorm=1.),
                  loss="mse")
                  # metrics=["mse"])

    prepare_data = get_prepare_data(df, params["timesteps"]) #partial function

    es = EarlyStopping(monitor="val_loss", patience=5,
                       min_delta=params["min_delta"] if "min_delta" in params else 0.0001,
                       restore_best_weights=True)

    nni_callback = LambdaCallback(on_epoch_end= lambda epoch,logs : nni.report_intermediate_result(logs["val_loss"]))

    history = model.fit_generator(prepare_data(mode="train"),
                                  epochs = 250,
                                  steps_per_epoch= L-params["timesteps"],
                                  validation_data = prepare_data(mode="eval"),
                                  validation_steps = len(df) - L,
                                  callbacks=[nni_callback, es],
                                  verbose=0
                                 )

    nni.report_final_result(np.min(history.history['val_loss']))