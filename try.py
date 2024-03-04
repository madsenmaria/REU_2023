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
from keras import regularizers
import random



# when splitting data, I want the option to randomize the year, and that case does a 20-20-60 -> or smthg

#HERE 1
x = list(range(1983,2019))
random.shuffle(x)

for i,val in enumerate(x):
    x[i] = [str(val)+'-11-01', str(val+1)+'-05-31']

train_range = x[:-6]
test_range=x[-6:-3]
val_range=x[-3:]


data_path = '../../../scratch/esalm/data/edata_tot.nc'
pph_tor = '../../../scratch/esalm/data/pper_tor_1979_2019.nc'
pph_hail='../../../scratch/esalm/data/pper_hail_1979_2019.nc'


s = splits.Train_Test_Val_Split(window=1, horizon=7, data_path=data_path,  pph_path_tor=pph_tor, pph_path_hail=pph_hail,
                                train_range=train_range, test_range=test_range, val_range=val_range)

data = s.split()


###

print(np.shape(data[1]))

def UNet_build(input, activation, depth, conv2d_depth, start_val):
    y = input
    skip_arr=[]
    #setting to correct aspect ratio
    y = tf.keras.layers.Conv3D(start_val, 3, strides=(1,1,3), activation='relu', input_shape=y.shape, 
                               activity_regularizer = regularizers.L2(0.01), padding='same')(y)

     
    y = layers.MaxPooling3D((1,1,1))(y)
 

    #contraction loop
    for i in range(depth+1):
        for j in range(conv2d_depth):
            y = tf.keras.layers.Conv3D(start_val*(2**(i-1)), 3, activation=activation, input_shape=y.shape, 
                                       activity_regularizer = regularizers.L2(0.01), padding='same')(y)
            if j == conv2d_depth-1:
                skip_arr.append(y)
        y = layers.MaxPooling3D((1,2,2))(y)



    end_val = start_val*(2**(depth))
    y = tf.keras.layers.Conv3D(end_val*2, 3, activation=activation, input_shape=y.shape, 
                               activity_regularizer = regularizers.L2(0.01), padding='same')(y)

    end_val=end_val*2

    #expansion loop
    for i in range(depth+2):
        for j in range(conv2d_depth):
            y = tf.keras.layers.Conv3D(end_val/(2**(i+1)), 3, activation=activation, activity_regularizer = regularizers.L2(0.01),
                                        input_shape=y.shape, padding='same')(y)

            # adding skip
            if j==0 and i>=1:
                sk = skip_arr.pop()
                y = tf.concat([sk,y], 4)
        y = layers.UpSampling3D((1,2,2))(y)
    

    #get to output shape
    y = tf.keras.layers.Conv3D(4, 3, activation=activation, input_shape=y.shape, 
                               activity_regularizer = regularizers.L2(0.001), padding='same')(y)

    print('pre')
    print(y.shape)


    return tf.math.reduce_mean(y, axis=1)


input = keras.Input(shape=(int(1), 32, 144, 5), name="full_input")

params={'activation': 'relu',
 'conv3d_depth': 2, #forever constant
 'start_val': 4, 
 'learning_rate': 0.001,
 'metrics': 'mse',
 'window':1,
 'depth':3, #forever contant
 }

x = UNet_build(input,
                activation = 'relu', 
                depth = params['depth'], 
                conv2d_depth = params['conv3d_depth'], 
                start_val = params['start_val'])
output = x



model = keras.Model(
    inputs = input, 
    outputs = output, 
    name='modelr')



# Compile the model with appropriate optimizer, loss, and metrics
opt = keras.optimizers.Adam(
    learning_rate = 0.001, amsgrad = False)

# Bind the model to the optimizer
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print('data[3] value counts')
print(pd.Series(np.ravel(np.sum(np.array(data[3], dtype='int64'), axis =-1))).value_counts())

predictions = model.predict(np.array(data[2], dtype='int64'))
print(pd.Series(np.ravel(np.sum(predictions, axis=-1))).value_counts())

# print('input check')
# print(np.sum(np.isnan(np.array(data[0], dtype='int64'))))
# print(np.sum(np.isinf(np.array(data[0], dtype='int64'))))

# print('actual')
# print(np.sum(np.isnan(np.array(data[1], dtype='int64'))))
# print(np.sum(np.isinf(np.array(data[1], dtype='int64'))))


# print('predictions check')
# print(np.sum(np.isnan(predictions)))
# print(np.sum(np.isinf(predictions)))

# print('data[3] shape')
# print(np.shape(data[3]))

scce = tf.keras.losses.SparseCategoricalCrossentropy()
loss = scce(np.array(data[3], dtype='int64'), predictions).numpy()
print(np.shape(loss))
print(pd.Series(np.ravel(np.sum(loss, axis=-1))).value_counts())
print(loss)
# print(np.sum(np.isnan(loss)))



#want the option to pick out certain dates for oversamplining

#want to be able to smooth out time (example 3-days)
