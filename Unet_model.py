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

print('Categorical Cross Entropy for hail')

testing_params = {
    'type': 'weight_mxe_loss',
    'fss_weight': 10,
    'function_weight': 5,
    'function_xtra_weight': 6,
}


name='UNet_'+testing_params['type']+'_'+str(testing_params['fss_weight'])+'_'+str(testing_params['function_weight'])+'_'+str(testing_params['function_xtra_weight'])+'_try2'
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


metric_dict = {
    'mse':'mse',
    'mae':'mae',
    'mape':'mean_absolute_percentage_error',
    'msle':'mean_squared_logarithmic_error',
    'cosine':'cosine_similarity',
    'logcosh':'logcosh'
}

loss_dict ={
    'weight_mxe_loss': weighted_loss([testing_params['fss_weight'], testing_params['function_weight'], testing_params['function_xtra_weight']]),
    'weighted_sigma_loss':  weighted_sigma_loss([testing_params['fss_weight'], testing_params['function_weight'], testing_params['function_xtra_weight']]),
    'mse':'mse',
    'mae':'mae',
    'msle':'mean_squared_logarithmic_error',
}


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

train_range = x[:-6]
test_range=x[-6:-3]
val_range=x[-3:]


data_path = '../../../scratch/esalm/data/edata_tot.nc'
pph_tor = '../../../scratch/esalm/data/pper_tor_1979_2019.nc'
pph_hail='../../../scratch/esalm/data/pper_hail_1979_2019.nc'

s = splits.Train_Test_Val_Split(window=30, horizon=7, data_path=data_path,  pph_path_tor=pph_tor, pph_path_hail=pph_hail,
                                train_range=train_range, test_range=test_range, val_range=val_range)

data = s.split()


### Building model
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
    y = tf.keras.layers.Conv3D(4, 3, activation=activation, input_shape=y.shape, padding='same')(y)

    return tf.math.reduce_mean(y, axis=1)


def model_maker(): 
    input = keras.Input(shape=(int(params['window']), 32, 144, 5), name="full_input")
    print('input defined shape')
    print(tf.shape(input))
    x = UNet_build(input, 
                   activation = params['activation'], 
                   depth = params['depth'], 
                   conv2d_depth = params['conv3d_depth'], 
                   start_val = params['start_val'])
    
    output = x
    # output = layers.Dense(1, params['activation'])(x)

    # output = tf.math.reduce_mean(output, axis=1)
    
    print(tf.shape(output))

    model = keras.Model(
        inputs = [input], 
        outputs = output, 
        name='modelr')
    
    # Compile the model with appropriate optimizer, loss, and metrics
    opt = keras.optimizers.experimental.SGD(
        learning_rate = params['learning_rate'])
    
    # Bind the model to the optimizer
    model.compile(optimizer=opt, 
                  metrics = metric_dict[params['metrics']], 
                  loss='categorical_crossentropy') #testing this right hereeee

    return model




model = model_maker()
early_stopping_cb = keras.callbacks.EarlyStopping(patience=10,
                                                  restore_best_weights=True,
                                                      min_delta=0.0001)

print('input actual size')
print(np.shape(data[0]))


h = model.fit(x=[data[0]], y=[data[1]], epochs=20, verbose=1,
                        validation_data=([data[2]], [data[3]]), 
                        steps_per_epoch = 20, batch_size=16)


# predictions = model.predict([data[4]])


### Graphinggggg

#Accuracy Plots

'''
print(np.shape(predictions))
print(np.shape(data[5][:,:,:,:]))
print(np.size(np.ravel(predictions[:,:,:,:,0])))
print(np.size(np.ravel(data[5][:,:,:,0])))

pred_tor = np.ravel(predictions[:,:,:,:,0])
pred_hail = np.ravel(predictions[:,:,:,:,1])
actual_tor = np.ravel(data[5][:,:,:,0])
actual_hail = np.ravel(data[5][:,:,:,0])


plt.rcParams['savefig.facecolor'] = "0.8"

fig = plt.figure()

ax2 = fig.add_subplot(2,2,1)
ax2 = plt.scatter(actual_tor, pred_tor,s=1)
ax2 = plt.title("Tornado")
ax2 = plt.xlabel("Actual Value")
ax2 = plt.ylabel("Predicted Value")
ax2 = plt.ylim([0,40])
ax2 = plt.plot([0,40], [0,40], color="black")


ax2 = fig.add_subplot(2,2,2)
ax2 = plt.scatter(actual_hail, pred_hail,s=1)
ax2 = plt.title("Hail")
ax2 = plt.xlabel("Actual Value")
ax2 = plt.ylabel("Predicted Value")
ax2 = plt.ylim([0,40])
ax2 = plt.plot([0,40], [0,40], color="black")

plt.tight_layout()
plt.savefig('./UNET_plots/'+name+'.png', transparent=True)



#### Hail Distribution 
fig = plt.figure()
nbins = 40

ax1 = fig.add_subplot(2,2,1)
hail_min = np.min(pred_hail)
hail_max = np.max(pred_hail)

hist, bin_edges = np.histogram(pred_hail, range=(hail_min,hail_max),bins=nbins)
bin_cents=(bin_edges[0:-1]+bin_edges[1:])/2
ax1 = plt.hist(pred_hail, range=(hail_min,hail_max), edgecolor='black',bins=nbins)
ax1 = plt.title("Hail Probability Predicted Distribution")
ax1 = plt.xlabel("hail prob")
ax1 = plt.ylabel("counts")


ax2 = fig.add_subplot(2,2,2)
hail_min = np.min(actual_hail)
hail_max = np.max(actual_hail)

hist, bin_edges = np.histogram(actual_hail, range=(hail_min,hail_max),bins=nbins)
bin_cents=(bin_edges[0:-1]+bin_edges[1:])/2
ax2 = plt.hist(actual_hail, range=(hail_min,hail_max), edgecolor='black',bins=nbins);
ax2 = plt.title("Hail Probability Actual Distribution")
ax2 = plt.xlabel("hail prob")
ax2 = plt.ylabel("counts")
plt.tight_layout()
plt.savefig('./UNET_plots/hail_hist_'+name+'.png', transparent=True)

### Tornado Distribution
fig = plt.figure()

ax1 = fig.add_subplot(2,2,1)
hail_min = np.min(actual_tor)
hail_max = np.max(actual_tor)

hist, bin_edges = np.histogram(pred_tor, range=(hail_min,hail_max),bins=nbins)
bin_cents=(bin_edges[0:-1]+bin_edges[1:])/2
ax1 = plt.hist(pred_tor, range=(hail_min,hail_max), edgecolor='black',bins=nbins)
ax1 = plt.title("Tor Probability Predicted Distribution")
ax1 = plt.xlabel("tor prob")
ax1 = plt.ylabel("counts")


ax2 = fig.add_subplot(2,2,2)

hist, bin_edges = np.histogram(actual_tor, range=(hail_min,hail_max),bins=nbins)
bin_cents=(bin_edges[0:-1]+bin_edges[1:])/2
ax2 = plt.hist(actual_tor, range=(hail_min,hail_max), edgecolor='black',bins=nbins);
ax2 = plt.title("Tor Probability Actual Distribution")
ax2 = plt.xlabel("tor prob")
ax2 = plt.ylabel("counts")
plt.tight_layout()
plt.savefig('./UNET_plots/tor_hist_'+name+'.png', transparent=True)


pph_tor = xr.load_dataset(pph_tor)
pph_hail = xr.load_dataset(pph_hail)


print(np.shape(pph_tor.lat))
print(np.shape(pph_tor.lon))

tor = predictions[:,:,:,:,0]
hail = predictions[:,:,:,:,1]

lat_data = np.zeros((64,96))
lat_data[:,:93]=pph_tor.lat.data[:64,:]

lon_data = np.zeros((64,96))
lon_data[:,:93]=pph_tor.lon.data[:64,:]

y_data = np.zeros(64)
y_data[:]=pph_tor.y.data[:64]

x_data = np.zeros(96)
x_data[:93]=pph_tor.x.data[:]



df_era = xr.Dataset(
    data_vars=dict(
        tor=(["time", "y", "x"], np.reshape(tor, (635, 64, 96))),
        hail=(["time", "y", "x"], np.reshape(hail, (635, 64, 96))),
        lat = (["y", "x"], lat_data),
        lon = (["y", "x"], lon_data)
    ),
    coords=dict(
        x=x_data,
        y=y_data,
        time=list(range(1,636)),
    ),
    attrs=dict(description="Attempt"),
)


df_era.to_netcdf('./'+name+'out_Unet_data.nc')
'''
