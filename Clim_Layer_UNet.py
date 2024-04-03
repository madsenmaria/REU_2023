from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
import train_test_val_split_Unet as splits
from tensorflow.keras import backend as K
import climate_split

#custom loss function using fss and m6e
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

        return (class_weights[0]*fss+class_weights[1]*mxe)

    return loss


#custom loss function parameters for ease
testing_params = {
    'fss_weight' : 20,
    'function_weight' : 20,
    'function_xtra_weight': 6,

}

#U-NET Parameters
params={'activation': 'relu',
 'conv3d_depth': 2, #forever constant
 'start_val': 4, 
 'learning_rate': 0.001,
 'metrics': 'mae',
 'loss': weighted_loss([testing_params['fss_weight'], testing_params['function_weight'], testing_params['function_xtra_weight']]), 
 'window':30,
 'depth':3, #forever contant
 }


### Prepping Data
data_path = '../../../ourdisk/hpc/ai2es/esalm/data/edata_tot.nc'
pph_hail='../../../ourdisk/hpc/ai2es/esalm/data/pper_hail_1979_2019.nc'
pph_tor='../../../ourdisk/hpc/ai2es/esalm/data/pper_tor_1979_2019.nc'

s = splits.Train_Test_Val_Split(window=30, horizon=7, data_path=data_path,  pph_path_tor=pph_tor, pph_path_hail=pph_hail)

data = s.split()



### Building model
def UNet_build(input, clim_in, activation, depth, conv2d_depth, start_val):
    y = input
    skip_arr=[]

    #setting input to correct aspect ratio
    y = tf.keras.layers.Conv3D(start_val-2, 3, strides=(1,1,3), activation='relu', input_shape=y.shape, padding='same')(y)
    y = layers.MaxPooling3D((params['window'],1,1))(y)

    #creating climatology
    clim = tf.keras.layers.Conv3D(2, 3, strides=(1,1,1), activation='relu', input_shape=clim_in.shape, padding='same')(clim_in)
    clim = layers.MaxPooling3D((1,2,2))(clim) #resize to be in correct format

    #combining input w climatology
    y = keras.layers.Concatenate(axis = -1)([y, clim]) 

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

            # adding skip layer
            if j==0 and i>=1:
                sk = skip_arr.pop()
                y = tf.concat([sk,y], 4)
        y = layers.UpSampling3D((1,2,2))(y)


    #get to output shape
    y = tf.keras.layers.Conv3D(2, 3, activation=activation, input_shape=y.shape, padding='same')(y)

    return y

def model_maker(): 
    input = keras.Input(shape=(int(params['window']), 32, 144, 5), name="full_input")
    clim_input = keras.Input(shape=(1, 64, 96, 2), name="input_cl")
    
    #creating U-NET
    x = UNet_build(input, clim_input, 
                   activation = params['activation'], 
                   depth = params['depth'], 
                   conv2d_depth = params['conv3d_depth'], 
                   start_val = params['start_val'])
    output = x

    #creating model
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
                                                      min_delta=0.0001)

clim = climate_split.climate_data()

h = model.fit(x=[data[0]], y=[data[1], clim[0]], epochs=30, verbose=1,
                        validation_data=([data[2]], [data[3], clim[1]]), 
                        steps_per_epoch = 20, batch_size=3, callbacks=early_stopping_cb)

print(h)

