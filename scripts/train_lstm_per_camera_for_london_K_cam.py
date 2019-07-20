#!/usr/bin/env python
# coding: utf-8
import itertools
import os
import sys
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"]="-1" #set to -1 to disable gpu (on fresh kernel)

import pandas as pd
import numpy as np

import keras.backend as K
from sklearn.metrics import pairwise_distances

from keras import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import Adam
from keras import regularizers


from keras.callbacks import EarlyStopping

from matplotlib import pyplot as plt

BATCH_SIZE = 1

# df must be pivoted df not the original one.
def get_data_generator(df, camera_index, K, TIMESTEPS, mode="train"):
    if mode == "train":
        while True:
            for i in range(0, L - TIMESTEPS):  # first 3/4th of the time (30 days)
                X = np.reshape(df.iloc[i:i + TIMESTEPS, closest_cam_indices[camera_index][:K]].values,
                               newshape=(BATCH_SIZE, TIMESTEPS, K))
                y = np.reshape(df.iloc[i + TIMESTEPS, camera_index], newshape=(
                BATCH_SIZE, 1))  # it is just one number but we reshape it as two dim tensor (1,1)
                yield (X, y)

    elif mode == "eval":
        while True:
            for i in range(L - TIMESTEPS, len(df) - TIMESTEPS):  # remaining quarter of data (10 days)
                X = np.reshape(df.iloc[i:i + TIMESTEPS, closest_cam_indices[camera_index][:K]].values,
                               newshape=(BATCH_SIZE, TIMESTEPS, K))
                y = np.reshape(df.iloc[i + TIMESTEPS, camera_index],
                               newshape=(BATCH_SIZE, 1))  # it is just one number but we reshape
                yield (X, y)

    elif mode == "predict":
        while True:
            for i in range(L - TIMESTEPS, len(df) - TIMESTEPS):  # same as eval without target y
                yield np.reshape(df.iloc[i:i + TIMESTEPS, closest_cam_indices[camera_index][:K]].values,
                                 newshape=(BATCH_SIZE, TIMESTEPS, K))


def get_model_simple(params):
    model = Sequential()
    model.add(LSTM(units=params["lstm_units"],
                   stateful=True,
                   input_shape=(params["timesteps"], params["K"]),
                   batch_size=params["batch_size"],
                   dropout=params["lstm_dropout"],
                   recurrent_dropout=params["lstm_recurrent_dropout"]))
    model.add(Dense(units=1,
                    activation="sigmoid",
                    activity_regularizer=regularizers.l2(l=params["dense_l2reg"])))

    return model


if __name__ == "__main__":

    print(K.tensorflow_backend._get_available_gpus())
    # tf.test.is_gpu_available()

    if len(sys.argv) != 2:
        sys.exit("usage: python thisfile.py <camera_index>")

    os.chdir(r"../immediate_results")

    df = pd.read_csv(r"cleandf_40days_930_to_1830_lin_interpolated.csv")

    onedf = df.groupby("CAMERA_ID").first()

    distances = pairwise_distances(onedf[["GEO_LON","GEO_LAT"]])

    closest_cam_indices = {}
    for i in range(distances.shape[0]):
        closest_cam_indices[i] = np.argsort(distances[i,:])

    # df.DENSITY_VALUE = np.log(df.DENSITY_VALUE)
    df.DENSITY_VALUE -= df.DENSITY_VALUE.min()
    df.DENSITY_VALUE /= df.DENSITY_VALUE.max() # scaling to [0,1]
    df = df.pivot(index="TIMESTAMP", columns="CAMERA_ID", values="DENSITY_VALUE")
    L = int(0.75 * len(df)) #3/4 of the length of data (over time) to be used as train and what ever comes after this would be for evaluation

    params = {}

    '''
    Parameters here correspond to best one found by NNI for camera 0  
    '''
    params["batch_size"] = 1
    params["camera_index"] = 0
    params["timesteps"] = 2
    params["K"] = 5
    params["lstm_units"] = 135
    params["lstm_dropout"] = 0.03216795095261028
    params["lstm_recurrent_dropout"] = 0.4885023555904258
    params["dense_l2reg"] = 0.0002523576467422392

    params["K"] = 1 #changed to see the average not using neighbors



    CURRENT_TIME = datetime.now().strftime("%Y-%m-%d %H-%M")

    for cam_index in [3]:

        model = get_model_simple(params)

        model.compile(optimizer=Adam(lr=0.001,clipnorm=1.), loss="mse")

        # print(model.summary())

        es = EarlyStopping(monitor="val_loss", patience=10, min_delta=0.00005, restore_best_weights=True)

        train_gen = get_data_generator(df, cam_index, params["K"], params["timesteps"], mode = "train")

        val_gen = get_data_generator(df, cam_index, params["K"], params["timesteps"], mode="eval")

        history = model.fit_generator(train_gen,
                                      epochs = 250,
                                      steps_per_epoch= L-params["timesteps"],
                                      validation_data = val_gen,
                                      validation_steps = len(df) - L,
                                      callbacks=[es],
                                      verbose = 2
                                     )

        one_test = itertools.islice(get_data_generator(df, cam_index, params["K"],
                                    params["timesteps"], mode="eval"), len(df) - L)
        a = list(one_test)
        X, y = zip(*a)
        X = np.array(X).reshape( (len(df)-L, params["timesteps"], params["K"]) )
        y = np.array(y).reshape( (len(df)-L, 1) )
        print(X.shape,y.shape)
        predictions = model.predict(X,batch_size=1)
        plt.plot(y)
        plt.plot(predictions)

        # np.save("evalX_with5_neighbors.npy",X)
        # np.save("evalY_with5_neighbors.npy",y)
        plt.savefig("lstm_cam_{}_no_neighbors.png".format(cam_index))
        plt.show()

        # print(cam_index,np.mean(np.square(y - predictions)),file=open("lstm_per_cam_no_neighbors_results.txt",'a'))
        print(cam_index,np.mean(np.square(y - predictions)))

        del model
        del history
        K.clear_session() #to hopefully prevent slow down after a few models have run..
