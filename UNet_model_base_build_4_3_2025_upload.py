import numpy as np
import xarray as xr
import pandas as pd
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
# import train_test_val_split_Unet as splits
import train_test_val_split_Unet_loc_data as splits
from tensorflow.keras import backend as K
import plotter
# import shap.shap as shap
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

# import sys

# num_1 = sys.argv[1]

testing_params = {
    'fss_weight' : 25, # switch every 7,
    'function_weight' : 30, # switch every 35,
    'function_xtra_weight': 6 #switch every time, # trying 8 for shits and giggles

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

#def UNet_build(input, clim_in,  teleconnections, activation, depth, conv2d_depth, start_val):


def UNet_build(input, clim_in, activation, depth, conv2d_depth, start_val):
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

    # # adding the teleconnections
    # tele =  tf.keras.layers.Conv3D(4, 3, strides=(1,1,1), activation='relu', input_shape=teleconnections.shape, padding='same')(teleconnections)
    # print("print tele shape: "+str(tele.shape))
    # y = keras.layers.Concatenate(axis = -1)([y, tele]) 


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
'''
# def UNet_build(input, clim_in, activation, depth, conv2d_depth, start_val):
#     y = input
#     skip_arr=[]
#     #setting to correct aspect ratio
#     y = tf.keras.layers.Conv3D(start_val, 3, strides=(1,1,3), activation='relu', input_shape=y.shape, padding='same')(y)
#     y = layers.MaxPooling3D((params['window'],1,1))(y)

#     # #creating climatology
#     clim = tf.keras.layers.Conv3D(2, 3, strides=(1,1,1), activation='relu', input_shape=clim_in.shape, padding='same')(clim_in)
#     clim = layers.MaxPooling3D((1,2,2))(clim)

#     # #combining w climatology
#     y = keras.layers.Concatenate(axis = -1)([y, clim]) 

#     # contraction loop
#     for i in range(depth+1):
#         for j in range(conv2d_depth):
#             y = tf.keras.layers.Conv3D(start_val*(2**(i-1)), 3, activation=activation, input_shape=y.shape, padding='same')(y)
#             if j == conv2d_depth-1:
#                 skip_arr.append(y)
#         y = layers.MaxPooling3D((1,2,2))(y)

    
#     end_val = start_val*(2**(depth))
#     y = tf.keras.layers.Conv3D(end_val*2, 3, activation=activation, input_shape=y.shape, padding='same')(y)

#     end_val=end_val*2

#     # expansion loop
#     for i in range(depth+2):
#         for j in range(conv2d_depth):
#             y = tf.keras.layers.Conv3D(end_val/(2**(i+1)), 3, activation=activation, input_shape=y.shape, padding='same')(y)

#             # adding skip
#             if j==0 and i>=1:
#                 sk = skip_arr.pop()
#                 y = tf.concat([sk,y], 4)
#         y = layers.UpSampling3D((1,2,2))(y)

#     # get to output shape
#     y = tf.keras.layers.Conv3D(2, 3, activation=activation, input_shape=y.shape, padding='same')(y)

#     return y
'''

def model_maker(): 
    input = keras.Input(shape=(int(params['window']), 32, 144, 6), name="full_input")
    clim_input = keras.Input(shape=(1, 64, 96, 1), name="input_cl")
    #tele = keras.Input(shape=(1, 2, 3, 4), name="tele") # no mjo
    
    x = UNet_build(input, clim_input,# tele,
                   activation = params['activation'], 
                   depth = params['depth'], 
                   conv2d_depth = params['conv3d_depth'], 
                   start_val = params['start_val'])
    output = x

    # model = keras.Model(
    #     inputs = [input, clim_input, tele], 
    #     outputs = output, 
    #     name='modelr')

    model = keras.Model(
        inputs = [input, clim_input], 
        outputs = output, 
        name='modelr')
    
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

data_0 = data[0].reshape(data[0].shape[0], int(params['window']), 32, 144, 6)
data_1 = data[1].reshape(data[1].shape[0], 7, 64, 96, 1) # double check
data_2 = data[2].reshape(data[2].shape[0], int(params['window']), 32, 144, 6)
data_3 = data[3].reshape(data[3].shape[0], 7, 64, 96, 1) # double check
data_m3 = np.reshape(data[6], (np.shape(data[6])[0], 1, 64, 96, 1))
data_m2 = np.reshape(data[7], (np.shape(data[7])[0], 1, 64, 96, 1))
# data_tele1 = np.reshape(data[-3], (np.shape(data[-3])[0], 1, 2, 3, 4)) # no mjo
# data_tele2 = np.reshape(data[-2], (np.shape(data[-2])[0], 1, 2, 3, 4))


# h = model.fit(x=[data_0, data_m3, data_tele1], y=data_1, epochs=30, verbose=1, 
#               validation_data=([data_2, data_m2, data_tele2], data_3), steps_per_epoch = 10, batch_size=2)

h = model.fit(x=[data_0, data_m3], y=data_1, epochs=30, verbose=1, 
              validation_data=([data_2, data_m2], data_3), steps_per_epoch = 10, batch_size=2)

# print(h)
# # # model.save('./sls_model.keras')
# data_4 = tf.convert_to_tensor(data[4].reshape(data[4].shape[0], int(params['window']), 32, 144, 6))
# data_m1 = tf.convert_to_tensor(np.reshape(data[-1], (np.shape(data[-1])[0], 1, 64, 96, 1)))



# # ## Saliency Plotting

data_4 = data[4].reshape(data[4].shape[0], int(params['window']), 32, 144, 6)
data_5 = data[5].reshape(data[5].shape[0], 7, 64, 96, 1)
data_m1 = np.reshape(data[8], (np.shape(data[8])[0], 1, 64, 96, 1))
# data_tele3 = np.reshape(data[-1], (np.shape(data[-1])[0], 1, 2, 3, 4))


predictions = model.predict([data_4, data_m1]) #, data_tele3])


#model.save('./models_w_MJO/model_w_mjo'+num_1+'.keras')


# print('predictions')
# print(np.shape(predictions))



#saving data

pph_hail_df = xr.load_dataset(pph_hail)

lat_data = np.zeros((64,96))
lat_data[:,:93]=pph_hail_df.lat.data[:64,:]

lon_data = np.zeros((64,96))
lon_data[:,:93]=pph_hail_df.lon.data[:64,:]

y_data = np.zeros(64)
y_data[:]=pph_hail_df.y.data[:64]

x_data = np.zeros(96)
x_data[:93]=pph_hail_df.x.data[:]

print('pred mean: '+str(predictions.mean()))

test_range=['2016-11-01', '2019-05-31']
datetimes_range = pd.date_range(start=test_range[0], end=test_range[1], freq='D')
nr = list(range(212 - 27, 212+153)) + list(range(212+365 - 27, 212+365+153))
datetimes_range = np.delete(datetimes_range, nr)
datetimes_range = datetimes_range[:-27]


df_era = xr.Dataset(
    data_vars=dict(
        hail=(["time", "pred_time", "y", "x"], np.reshape(predictions[:, :, :, :], (555, 7, 64, 96))),
        lat = (["y", "x"], lat_data),
        lon = (["y", "x"], lon_data)
    ),
    coords=dict(
        x=x_data,
        y=y_data,
        time=datetimes_range.to_pydatetime(),
        pred_time = list(range(1,8))
    ),
    attrs=dict(description="Attempt"),
)


#df_era.to_netcdf('../../../scratch/esalm/models_w_mjo'+num_1+'.nc')




plotter.print_prob_of_detection(df_era, data[5], 0)
plotter.print_prob_of_detection(df_era, data[5], 5)
plotter.print_prob_of_detection(df_era, data[5], 10)
plotter.print_prob_of_detection(df_era, data[5], 15)

print('success')


# print(type(predictions))
# print(predictions.shape)
# # print(predictions.max(axis=(1,2,3,4)))
# print(predictions.max())
# pred_max_cond = predictions.max(axis=(1,2,3,4)) > 20

# data_5 = data[5].reshape(data[5].shape[0], 1, 64, 96, 1)
# act_max_cond = data_5.max(axis=(1,2,3,4)) > 20

# cond_full = np.all([pred_max_cond, act_max_cond], axis=0)
# cond_full2 = np.where(cond_full == True)[0]
# print(cond_full2)



# output_layer = model.output

# # Create a model that outputs the desired layer's predictions

# grad_model = Model(inputs=model.input, outputs=output_layer)


# # Get the model's predictions
# with tf.GradientTape() as tape:
#     tape.watch([data_4, data_m1])
#     preds = grad_model([data_4, data_m1])
#     print(preds.shape)

#     # Choose the class or channel of interest (e.g., first channel for binary segmentation)
#     class_idx = 0  # Change based on your use case
#     class_output = preds[:, :, :, :, :]

#     grads = tape.gradient(class_output, [data_4, data_m1])

#     # Compute the saliency map


#     with open('../../../scratch/esalm/era5_sal.npy', 'wb') as f:
#         np.save(f, grads[0])

#     with open('../../../scratch/esalm/clomo_sal.npy', 'wb') as f:
#         np.save(f, grads[1])



# # FUTURE ELEANOR: the fact that your using tor is really not that important bc its the same data format as hail

# pph_tor_df = xr.load_dataset(pph_tor)

# lat_data = np.zeros((64,96))
# lat_data[:,:93]=pph_tor_df.lat.data[:64,:]

# lon_data = np.zeros((64,96))
# lon_data[:,:93]=pph_tor_df.lon.data[:64,:]

# y_data = np.zeros(64)
# y_data[:]=pph_tor_df.y.data[:64]

# x_data = np.zeros(96)
# x_data[:93]=pph_tor_df.x.data[:]

# test_range=['2016-11-01', '2019-05-31']
# datetimes_range = pd.date_range(start=test_range[0], end=test_range[1], freq='D')
# nr = list(range(212, 212+153)) + list(range(212+365, 212+365+153)) 
# datetimes_range = np.delete(datetimes_range, nr)

# df_era = xr.Dataset(
#     data_vars=dict(
#         hail=(["time", "y", "x"], np.reshape(predictions[:, :, :, :], (636, 64, 96))),
#         lat = (["y", "x"], lat_data),
#         lon = (["y", "x"], lon_data)
#     ),
#     coords=dict(
#         x=x_data,
#         y=y_data,
#         time=datetimes_range.to_pydatetime(),
#     ),
#     attrs=dict(description="Attempt"),
# )

# df_era.to_netcdf('../../../scratch/esalm/saliency_pred.nc')


# with open('../../../scratch/esalm/era5_sal.npy', 'rb') as f:
#     era_sal = np.load(f)

# print(era_sal.max())

# ox = np.arange(90, 10, step=-2.5)
# oy = np.arange(-180, 180, step= 2.5)
# plt.contourf(oy, ox, era_sal[cond_full2[2], :, :, :, 1].mean(axis=0), cmap='RdBu')
# plt.savefig('./test_salnum1')


# print('success')


'''

# shap_values = explainer.shap_values(X.iloc[299, :], nsamples=500)
# shap.force_plot(explainer.expected_value, shap_values, X_display.iloc[299, :])

#### ------



# predictions = model.predict([data[4], np.reshape(data[-1], (np.shape(data[-1])[0], 1, 64, 96))])


# print('predictions')
# print(np.shape(predictions))


# #saving data

# pph_tor_df = xr.load_dataset(pph_tor)

# lat_data = np.zeros((64,96))
# lat_data[:,:93]=pph_tor_df.lat.data[:64,:]

# lon_data = np.zeros((64,96))
# lon_data[:,:93]=pph_tor_df.lon.data[:64,:]

# y_data = np.zeros(64)
# y_data[:]=pph_tor_df.y.data[:64]

# x_data = np.zeros(96)
# x_data[:93]=pph_tor_df.x.data[:]

# print(np.shape(predictions))

# test_range=['2016-11-01', '2019-05-31']
# datetimes_range = pd.date_range(start=test_range[0], end=test_range[1], freq='D')
# nr = list(range(212, 212+153)) + list(range(212+365, 212+365+153)) 
# datetimes_range = np.delete(datetimes_range, nr)


# df_era = xr.Dataset(
#     data_vars=dict(
#         hail=(["time", "y", "x"], np.reshape(predictions[:, :, :, :], (636, 64, 96))),
#         # hail=(["time", "y", "x"], np.reshape(predictions[:, :, :, :, 1], (636, 64, 96))),
#         lat = (["y", "x"], lat_data),
#         lon = (["y", "x"], lon_data)
#     ),
#     coords=dict(
#         x=x_data,
#         y=y_data,
#         time=datetimes_range.to_pydatetime(),
#     ),
#     attrs=dict(description="Attempt"),
# )

# plotter.print_prob_of_detection(df_era, data[5], 'outfile')
# plotter.print_false_detection(df_era, data[5], 'outfile')
# plotter.print_csi(df_era, data[5], 'outfile')
'''
'''
df_era.to_netcdf('../../../scratch/esalm/out_data/resample_test_Unet_data_10_15_6.nc')

# plotter.plot_prob_of_detection(hail_predictions, test_pph, outfile_name)


### Graphinggggg

#Accuracy Plots

# '''
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
