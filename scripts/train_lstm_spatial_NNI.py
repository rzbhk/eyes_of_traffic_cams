#!/usr/bin/env python
# coding: utf-8



import os
# import io
# from time import time
# from datetime import datetime
import itertools
os.environ["CUDA_VISIBLE_DEVICES"]="-1" #set to -1 to disable gpu (on fresh kernel)


import keras
import keras.backend as K


# K.tensorflow_backend._get_available_gpus()
# tf.test.is_gpu_available()


import pandas as pd
import numpy as np

import nni


from keras import Sequential
from keras.layers import Dense, Activation, LSTM
from keras.optimizers import Adam, RMSprop
from keras import regularizers

# from keras.layers import Input, Concatenate, Reshape, Lambda, Dropout
from keras.layers import TimeDistributed, Bidirectional
# from keras.models import Model

from keras.callbacks import TensorBoard, LambdaCallback
from keras.callbacks import EarlyStopping, ModelCheckpoint
# from keras.utils import plot_model


BATCH_SIZE = 1


def get_data_generator_N_to_M(df, input_cameras_index_list, target_cameras_index_list, timesteps, mode="train"):
    if mode == "train":
        while True:
            for i in range(0, L - timesteps):
                X = np.reshape(df.iloc[i:i + timesteps, input_cameras_index_list].values,
                               newshape=(BATCH_SIZE, timesteps, len(input_cameras_index_list)))
                y = np.reshape(df.iloc[i + timesteps, target_cameras_index_list].values,
                               newshape=(BATCH_SIZE, len(target_cameras_index_list)))
                yield (X, y)

    elif mode == "eval":
        while True:
            for i in range(L - timesteps, len(df) - timesteps):
                X = np.reshape(df.iloc[i:i + timesteps, input_cameras_index_list].values,
                               newshape=(BATCH_SIZE, timesteps, len(input_cameras_index_list)))
                y = np.reshape(df.iloc[i + timesteps, target_cameras_index_list].values,
                               newshape=(BATCH_SIZE, len(target_cameras_index_list)))
                yield (X, y)


def get_model(params):
    model = Sequential()

    lstm_layer = LSTM(units=params["lstm_units"],
                        stateful=True,
                        batch_size=params["batch_size"],
                        input_shape=(params["timesteps"], params["input_N"]),
                        recurrent_dropout=params["lstm_recurrent_dropout"])

    if "bidirectional" in params and params["bidirectional"] == True:
        lstm_layer = Bidirectional(lstm_layer,
                                   batch_input_shape=(params["batch_size"], params["timesteps"], params["input_N"]) )

    model.add(lstm_layer)

    model.add(Dense(units=params["output_M"], activation="sigmoid",
                    activity_regularizer=regularizers.l2(l=params["dense_l2reg"])))

    return model

if __name__ == "__main__":

    os.chdir(r"../immediate_results")

    df = pd.read_csv(r"cleandf_40days_930_to_1830_lin_interpolated.csv")

    # onedf = df.groupby("CAMERA_ID").first()

    # from sklearn.metrics import pairwise_distances

    # distances = pairwise_distances(onedf[["GEO_LON","GEO_LAT"]])

    # closest_cam_indices = {}
    # for i in range(distances.shape[0]):
    #     closest_cam_indices[i] = np.argsort(distances[i,:])


    # df.DENSITY_VALUE = np.log(df.DENSITY_VALUE)
    df.DENSITY_VALUE -= df.DENSITY_VALUE.min()
    df.DENSITY_VALUE /= df.DENSITY_VALUE.max() # scaling to [0,1]
    df = df.pivot(index="TIMESTAMP",columns="CAMERA_ID",values="DENSITY_VALUE")
    L = int(0.75 * len(df)) #3/4 of the length of data (over time) to be used as train and what ever comes after this would be for evaluation

    NUM_CAMERAS = len(df.columns)

    # INPUT_CAMERAS_LENS = [1, 2, 4, 8, 16, 32]

    # CURRENT_TIME = datetime.now().strftime("%Y-%m-%d %H-%M")

    params = nni.get_next_parameter()

    params["batch_size"] = 1
    params["timesteps"] = int(params["timesteps"])
    params["lstm_units"] = int(params["lstm_units"])
    params["input_N"] = int(params["input_N"])
    params["random_seed"] = int(params["random_seed"])

    np.random.seed(params["random_seed"])

    in_list = np.random.choice(range(NUM_CAMERAS),size = params["input_N"], replace= False).tolist()
    out_list = [x for x in range(NUM_CAMERAS) if x not in in_list]
    params["output_M"]=len(out_list)

    model = get_model(params)

    model.compile(optimizer=Adam(lr=0.001,clipnorm=1.), loss="mse")#, metrics=["mape"])

    print(model.summary())
#   plot_model(model, to_file='model.png', show_shapes=True)

    es = EarlyStopping(monitor="val_loss", patience=10, min_delta=0.0001, restore_best_weights=True)

    train_gen = get_data_generator_N_to_M(df,
                                          in_list,
                                          out_list,
                                          params["timesteps"],
                                          mode = "train")
    #don't reuse generator as it might be messed up.

    val_gen = get_data_generator_N_to_M(df,
                                          in_list,
                                          out_list,
                                          params["timesteps"],
                                          mode = "eval")

    one_test = itertools.islice(val_gen, len(df) - L) #it should be fine as it is one whole eval epoch

    nni_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: nni.report_intermediate_result(logs["val_loss"]))

    history = model.fit_generator(train_gen,
                                  epochs = 250,
                                  steps_per_epoch= L-params["timesteps"],
                                  validation_data = val_gen,
                                  validation_steps = len(df) - L,
                                  callbacks=[es, nni_callback]
                                 )

    a = list(one_test)
    X, y = zip(*a)
    X = np.array(X).reshape( (len(df)-L, params["timesteps"], params["input_N"]) )
    y = np.array(y).reshape( (len(df)-L, params["output_M"]) )
    # print(X.shape,y.shape)

    predictions = model.predict(X, batch_size=1)

    mses = []
    for i in range(y.shape[1]):
        mses.append(np.mean(np.square(y[:,i] - predictions[:,i])))

    nni.report_final_result(np.mean(mses))#should it be best validation loss or evaluation mse? They are not
    # necessarily the same in multivariate output cases.

    # del model
    # del history
    # K.clear_session()  # to hopefully prevent slow down after a few models have run..

    # plt.plot(y[:,0])
    # plt.plot(predictions[:,0])
    # plt.title("camera_index_0 forecast vs actual spatial")
    # plt.show()
