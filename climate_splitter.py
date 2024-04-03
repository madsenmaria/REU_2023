import pandas as pd
import numpy as np
from scipy.io import loadmat
import xarray as xr
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from keras import regularizers
import random


def climate_data():

    hail_data = xr.load_dataset('../../../ourdisk/hpc/ai2es/esalm/data/pper_hail_1979_2019.nc')
    h = hail_data.p_perfect_hail

    yr = list(range(1983,2019))
    tot_data = np.zeros((36*212,65,93))
    for i in range(len(yr)-1):
        temp_t_slicer = slice(str(yr[i])+'-11-01', str(yr[i+1])+'-05-31')
        t = h.sel(time = temp_t_slicer)
        if np.shape(t)[0] > 212:
            tot_data[212*i:212*i+120,:,:] = t[:120, :, :]
            tot_data[212*i+120:212*i+212,:,:] = t[121:, :, :]
        else:
            tot_data[212*i:212*i+212,:,:] = t

    padded = np.pad(tot_data, pad_width = ((14, 0), (2, 0), (2,0)), mode='constant', constant_values = 0)

    shape = (15, 3, 3)
    d = np.mean(np.lib.stride_tricks.sliding_window_view(padded, shape), axis=(-3,-2,-1))


    yr = list(range(1983,2020))
    tot_time = np.array([])

    for i in range(len(yr)-1):
        temp_t_slicer = slice(str(yr[i])+'-11-01', str(yr[i+1])+'-05-31')
        t = h.sel(time = temp_t_slicer)
        label_list = t.time.values
        
        for val in label_list: 
            tot_time = np.append(tot_time, str(val).split('T')[0])
            

    drops = [120, 121+212*4, 122+212*8, 123+212*12, 124+212*16, 125+212*20, 126+212*24, 127+212*28, 128+212*32]
    drops.reverse()

    for drop in drops:
        tot_time = np.delete(tot_time, drop)


    tor_data = xr.load_dataset('../../../ourdisk/hpc/ai2es/esalm/data/pper_tor_1979_2019.nc')
    tr = tor_data.p_perfect_tor

    tor_data = np.zeros((36*212,65,93))
    for i in range(len(yr)-1):
        temp_t_slicer = slice(str(yr[i])+'-11-01', str(yr[i+1])+'-05-31')
        t = tr.sel(time = temp_t_slicer)
        if np.shape(t)[0] > 212:
            tor_data[212*i:212*i+120,:,:] = t[:120, :, :]
            tor_data[212*i+120:212*i+212,:,:] = t[121:, :, :]
        else:
            tor_data[212*i:212*i+212,:,:] = t

    padded = np.pad(tor_data, pad_width = ((14, 0), (2, 0), (2,0)), mode='constant', constant_values = 0)

    shape = (15, 3, 3)
    tro = np.mean(np.lib.stride_tricks.sliding_window_view(padded, shape), axis=(-3,-2,-1))


    df = xr.Dataset(
        data_vars=dict(
            hail=(["time", "y", "x"], d),
            tor = (["time", "y", "x"], tro),
        ),
        coords=dict(
            x=(["x"], h.x.values),
            y=(["y"], h.y.values),
            time=tot_time,
        ),

        attrs=dict(description="Attempt"),
    )


    climo_xr = df


    train_range=['1983-11-01', '2013-05-31']
    test_range=['2016-11-01', '2019-05-31']
    val_range=['2013-11-01', '2016-05-31']

    climo_xr_train = climo_xr.sel(time = slice(train_range[0], train_range[1]))
    climo_xr_test = climo_xr.sel(time = slice(test_range[0], test_range[1]))
    climo_xr_val = climo_xr.sel(time = slice(val_range[0], val_range[1]))


    # climo = np.zeros((np.shape(climo_xr.hail)[0], 64, 96))

    climo_train = np.zeros((np.shape(climo_xr_train.hail)[0], 64, 96, 2))
    climo_test = np.zeros((np.shape(climo_xr_test.hail)[0], 64, 96, 2))
    climo_val = np.zeros((np.shape(climo_xr_val.hail)[0], 64, 96, 2))


    climo_train[:, :, :93, 0] = climo_xr_test.hail.values[:,:64,:]
    climo_train[:, :, :93, 1] = climo_xr_test.tor.values[:,:64,:]

    climo_test[:, :, :93, 0] = climo_xr_test.hail.values[:,:64,:]
    climo_test[:, :, :93, 1] = climo_xr_test.tor.values[:,:64,:]

    climo_val[:, :, :93, 0] = climo_xr_val.hail.values[:,:64,:]
    climo_val[:, :, :93, 1] = climo_xr_val.tor.values[:,:64,:]


    ### -- THESE ARE FOR CATEGORICAL CROSS ENTROPY UNET CLIMATOLOGY -- ###

    # bins = np.array([0, 24, 24*2, 24*3])
    # climo[:, :, :93] = climo_xr.hail.values[:,:64,:]
    # climo = np.digitize(climo, bins)-1
    # climo = np.eye(4)[climo]
    

    return climo_train, climo_val, climo_test

