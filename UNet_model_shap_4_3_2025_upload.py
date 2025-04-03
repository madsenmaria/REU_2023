import numpy as np
import xarray as xr
import pandas as pd
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
import train_test_val_split_Unet_loc_data as splits
from tensorflow.keras import backend as K
import plotter
# import shap.shap as shap
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import shap.shap as shap


testing_params = {
    'fss_weight' : 25, # switch every 7,
    'function_weight' : 30, # switch every 35,
    'function_xtra_weight': 8 #switch every time, # trying 8 for shits and giggles

}


print(testing_params)

def weighted_loss(class_weights):
    def loss(target_tensor, prediction_tensor):

        # trying something stupid
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

        return class_weights[0]*fss+class_weights[1]*mxe

    return loss



metric_dict = {
    'mse':'mse',
    'mae':'mae',
    'mape':'mean_absolute_percentage_error',
    'msle':'mean_squared_logarithmic_error',
    'cosine':'cosine_similarity',
    'logcosh':'logcosh'
}

loss_dict = {
    'weight_mxe_loss': weighted_loss([testing_params['fss_weight'], testing_params['function_weight'], testing_params['function_xtra_weight']]),
    'mse':'mse',
    'mae':'mae',
    'msle':'mean_squared_logarithmic_error',
}


params = {'activation': 'relu',
 'conv3d_depth': 2, #forever constant
 'start_val': 6, 
 'learning_rate': 1e-5,
 'metrics': 'mae',
 'loss': loss_dict['weight_mxe_loss'], 
 'window':14,
 'depth':3, #forever contant
 }


### Doing data stuff


data_path ='../../../ourdisk/hpc/ai2es/esalm/data/era_cin_cape_data.nc'
pph_hail='../../../ourdisk/hpc/ai2es/esalm/data/pper_hail_1979_2019.nc'
pph_tor='../../../ourdisk/hpc/ai2es/esalm/data/pper_tor_1979_2019.nc'

s = splits.Train_Test_Val_Split(window=14, horizon=7, data_path=data_path,  pph_path_tor=pph_tor, pph_path_hail=pph_hail)

data = s.split()


### Building model

def UNet_build(input, clim_in,  teleconnections, activation, depth, conv2d_depth, start_val):
    y = input
    skip_arr=[]
    #setting to correct aspect ratio
    y = tf.keras.layers.Conv3D(start_val, 3, strides=(1,1,3), activation='relu', input_shape=y.shape, padding='same')(y)
    y = layers.MaxPooling3D((params['window'],1,1))(y)

    # #creating climatology
    clim = tf.keras.layers.Conv3D(start_val, 3, strides=(1,1,1), activation='relu', input_shape=clim_in.shape, padding='same')(clim_in)
    clim = layers.MaxPooling3D((1,2,2))(clim)

    # #combining w climatology
    y = keras.layers.Concatenate(axis = -1)([y, clim]) 

    # contraction loop
    for i in range(depth+1):
        for j in range(conv2d_depth):
            y = tf.keras.layers.Conv3D(int(start_val*(2**(i-1))), 3, activation=activation, input_shape=y.shape, padding='same')(y)
            if j == conv2d_depth-1:
                skip_arr.append(y)
        y = layers.MaxPooling3D((1,2,2))(y)

    print('didnt dye in this one')

    # adding the teleconnections
    tele =  tf.keras.layers.Conv3D(4, 3, strides=(1,1,1), activation='relu', input_shape=teleconnections.shape, padding='same')(teleconnections)
    print("print tele shape: "+str(tele.shape))
    y = keras.layers.Concatenate(axis = -1)([y, tele]) 


    end_val = start_val*(2**(depth))
    y = tf.keras.layers.Conv3D(end_val*2, 3, activation=activation, input_shape=y.shape, padding='same')(y)

    end_val=end_val*2

    # expansion loop
    for i in range(depth+2):
        for j in range(conv2d_depth):
            y = tf.keras.layers.Conv3D(int(end_val/(2**(i+1))), 3, activation=activation, input_shape=y.shape, padding='same')(y)

            # adding skip
            if j==0 and i>=1:
                sk = skip_arr.pop()
                y =  tf.keras.layers.Concatenate(axis=4)([sk, y])
        y = layers.UpSampling3D((1,2,2))(y)

    y = layers.UpSampling3D((7,1,1))(y) # added for the extra days

    # get to output shape
    y = tf.keras.layers.Conv3D(1, 3, activation=activation, input_shape=y.shape, padding='same')(y)

    return y


def model_maker(): 
    input = keras.Input(shape=(int(params['window']), 32, 144, 6), name="full_input")
    clim_input = keras.Input(shape=(1, 64, 96, 1), name="input_cl")
    tele = keras.Input(shape=(1, 2, 3, 4), name="tele")
    
    x = UNet_build(input, clim_input, tele,
                   activation = params['activation'], 
                   depth = params['depth'], 
                   conv2d_depth = params['conv3d_depth'], 
                   start_val = params['start_val'])

    full_shape = 0
    for shaper in x.shape:
        if shaper:
            full_shape = full_shape*shaper
    print('x shape: '+str(full_shape/(64*96)))
    
    output = tf.reshape(x, [-1, 64*96*7]) # mult by 7 for extra days predicting

    model = keras.Model(
        inputs = [input, clim_input, tele], 
        outputs = output, 
        name='modelr')

    print('im after model stuff')
    
    # Compile the model with appropriate optimizer, loss, and metrics
    opt = keras.optimizers.Adam(
        learning_rate = 0.001, amsgrad = False)
    
    # Bind the model to the optimizer
    model.compile(optimizer=opt, 
                  metrics = params['metrics'], 
                  loss=params['loss'])

    return model



model = model_maker()

early_stopping_cb = keras.callbacks.EarlyStopping(patience=10,
                                                  restore_best_weights=True,
                                                      min_delta=0.00001)

data_0 = data[0][:, :, :, :, :].reshape(data[0].shape[0], int(params['window']), 32, 144, 6)
data_1 = data[1].reshape(data[1].shape[0], 64*96*7)
data_2 = data[2][:, :, :, :, :].reshape(data[2].shape[0], int(params['window']), 32, 144, 6)
data_3 = data[3].reshape(data[3].shape[0], 64*96*7)
data_m3 = np.reshape(data[6], (np.shape(data[6])[0], 1, 64, 96, 1))
data_m2 = np.reshape(data[7], (np.shape(data[7])[0], 1, 64, 96, 1))
data_tele1 = np.reshape(data[-3], (np.shape(data[-3])[0], 1, 2, 3, 4))
data_tele2 = np.reshape(data[-2], (np.shape(data[-2])[0], 1, 2, 3, 4))



h = model.fit(x=[data_0, data_m3, data_tele1], y=data_1, epochs=30, verbose=1, 
              validation_data=([data_2, data_m2, data_tele2], data_3), steps_per_epoch = 10, batch_size=2)




data_4 = data[4][:, :, :, :, :].reshape(data[4].shape[0], int(params['window']), 32, 144, 6)
# data_5 = data[5].reshape(data[5].shape[0], 1, 64, 96, 1) # I think this is an unnecessary declaration

data_m1 = np.reshape(data[8], (np.shape(data[8])[0], 1, 64, 96, 1))
data_tele3 = np.reshape(data[-1], (np.shape(data[-1])[0], 1, 2, 3, 4))

X = [data_4[:2, :, :, :, :], data_m1[:2, :, :, :, :], data_tele3[:2, :, :, :, :]]

d42 = np.reshape(data_4[3, :, :, :, :], (1, int(params['window']), 32, 144, 6))
dm12 = np.reshape(data_m1[3, :, :, :, :], (1, 1, 64, 96, 1))
data_tele32 = np.reshape(data_tele3[3, :, :, :, :], (1, 1, 2, 3, 4))

Y = [d42, dm12, data_tele32]

unet_expl = shap.DeepExplainer((model.input, model.output) , X)
raw_shap_explanations = unet_expl.shap_values(Y, check_additivity=False)

with open('raw_shap_exp_test.npy', 'wb') as f:
    np.save(f, raw_shap_explanations[0])
    np.save(f, raw_shap_explanations[1])


print()

print(raw_shap_explanations)


print("type of raw_shap_explanations: "+str(type(raw_shap_explanations)))
print("should be shape of (2, 14, 32, 144, 6): "+str(np.shape(raw_shap_explanations[0])))
print("should be shape of (2, 14, 64, 96, 1)... maybe: "+str(np.shape(raw_shap_explanations[1])))


print('success')

