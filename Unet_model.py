import pandas as pd
import numpy as np
from scipy.io import loadmat
import xarray as xr
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
import train_test_val_split_Unet as splits
from tensorflow.keras import backend as K
import random

print('Categorical UNET')

testing_params = {
    'type': 'weight_mxe_loss',
    'fss_weight': 10,
    'function_weight': 5,
    'function_xtra_weight': 6,
}

# IGNORE
def weighted_loss(class_weights):
    def loss(target_tensor, prediction_tensor):

        bin_target = tf.where(target_tensor == 0, target_tensor, 1)
        bin_predict = tf.where(prediction_tensor == 0, prediction_tensor, 1)

        bin_mse = K.mean(
            (bin_target-bin_predict)**2
        )

        div = K.mean(
            bin_target**2+bin_predict**2
        )

        fss = (
            bin_mse/div
        )

        mxe = K.mean(
            (target_tensor-prediction_tensor)**class_weights[2]
        )

        return (class_weights[0]*fss+class_weights[1]*mxe)*5

    return loss

#IGNORE
def weighted_sigma_loss(class_weights):
    def loss(target_tensor, prediction_tensor):

        bin_target = tf.where(target_tensor == 0, target_tensor, 1)
        bin_predict = tf.where(prediction_tensor == 0, prediction_tensor, 1)
        
        x = abs(target_tensor-prediction_tensor)

        bin_mse = K.mean(
            (bin_target-bin_predict)**2
        )

        div = K.mean(
            bin_target**2+bin_predict**2
        )

        fss = (
            bin_mse/div
        )

        new_func = (
            100*(1/(1+2**(-x/class_weights[2])) - 1/2)
        )

        return class_weights[0]*fss+class_weights[1]*new_func

    return loss


#IGNORE
metric_dict = {
    'mse':'mse',
    'mae':'mae',
    'mape':'mean_absolute_percentage_error',
    'msle':'mean_squared_logarithmic_error',
    'cosine':'cosine_similarity',
    'logcosh':'logcosh'
}

#IGNORE
loss_dict ={
    'weight_mxe_loss': weighted_loss([testing_params['fss_weight'], testing_params['function_weight'], testing_params['function_xtra_weight']]),
    'weighted_sigma_loss':  weighted_sigma_loss([testing_params['fss_weight'], testing_params['function_weight'], testing_params['function_xtra_weight']]),
    'mse':'mse',
    'mae':'mae',
    'msle':'mean_squared_logarithmic_error',
}

#IGNORE
params={'activation': 'relu',
 'conv3d_depth': 2, #forever constant
 'start_val': 4, 
 'learning_rate': 0.001,
 'metrics': 'mse',
 'loss': testing_params['type'], 
 'window':30,
 'depth':3, #forever contant
 }


### Doing data stuff

x = list(range(1983,2019))
random.shuffle(x)

for i,val in enumerate(x):
    x[i] = [str(val)+'-11-01', str(val+1)+'-05-31']

train_range = x[:-4]
test_range=x[-6:-3]
val_range=x[-3:]


data_path = '../../../ourdisk/hpc/ai2es/esalm/data/edata_tot.nc'
pph_tor = '../../../ourdisk/hpc/ai2es/esalm/data/pper_tor_1979_2019.nc'
pph_hail='../../../ourdisk/hpc/ai2es/esalm/data/pper_hail_1979_2019.nc'

s = splits.Train_Test_Val_Split(window=30, horizon=7, data_path=data_path,  pph_path_tor=pph_tor, pph_path_hail=pph_hail,
                                train_range=train_range, test_range=test_range, val_range=val_range)

data = s.split()


### Building model -- START HERE
def UNet_build(input, activation, depth, conv2d_depth, start_val):
    y = input
    skip_arr=[]
    #setting to correct aspect ratio
    y = tf.keras.layers.Conv3D(start_val, 3, strides=(1,1,3), activation='relu', input_shape=y.shape, padding='same')(y)
    y = layers.MaxPooling3D((params['window'],1,1))(y)
    #contraction loop
    for i in range(depth+1):
        for j in range(conv2d_depth):
            y = tf.keras.layers.Conv3D(start_val*(2**(i-1)), 3, activation=activation, input_shape=y.shape, padding='same')(y)
            if j == conv2d_depth-1:
                skip_arr.append(y)
        y = layers.MaxPooling3D((1,2,2))(y)


    
    end_val = start_val*(2**(depth))
    y = tf.keras.layers.Conv3D(end_val*2, 3, activation=activation, input_shape=y.shape, padding='same')(y)

    end_val=end_val*2

    #expansion loop
    for i in range(depth+2):
        for j in range(conv2d_depth):
            y = tf.keras.layers.Conv3D(end_val/(2**(i+1)), 3, activation=activation, input_shape=y.shape, padding='same')(y)

            # adding skip
            if j==0 and i>=1:
                sk = skip_arr.pop()
                y = tf.concat([sk,y], 4)
        y = layers.UpSampling3D((1,2,2))(y)

    #get to output shape
    y = tf.keras.layers.Conv3D(2, 3, activation=activation, input_shape=y.shape, padding='same')(y)

    return y


def model_maker(): 
    input = keras.Input(shape=(int(params['window']), 32, 144, 5), name="full_input")
    x = UNet_build(input, 
                   activation = params['activation'], 
                   depth = params['depth'], 
                   conv2d_depth = params['conv3d_depth'], 
                   start_val = params['start_val'])
    
    output = x

    output = tf.math.reduce_mean(output, axis=1)

    model = keras.Model(
        inputs = [input], 
        outputs = output, 
        name='modelr')
    
    # Compile the model with appropriate optimizer, loss, and metrics
    opt = keras.optimizers.Adam(
        learning_rate = params['learning_rate'],
                                amsgrad = False)
    
    # Bind the model to the optimizer
    model.compile(optimizer=opt, 
                  metrics = metric_dict[params['metrics']], 
                  loss='categorical_crossentropy') #testing this right hereeee

    return model




model = model_maker()
early_stopping_cb = keras.callbacks.EarlyStopping(patience=10,
                                                  restore_best_weights=True,
                                                      min_delta=0.0001)

print(np.shape(data[1]))
h = model.fit(x=[data[0]], y=[data[1]], epochs=20, verbose=1,
                        validation_data=([data[2]], [data[3]]), 
                        steps_per_epoch = 20, batch_size=16)


