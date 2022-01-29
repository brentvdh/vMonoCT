#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Friday December 4 2020
"""

#pip install keras (x)
#pip install tensorflow (x)
#pip install numpy (x)
#pip install SimpleITK (x)
#pip install pydot

import os
import time
import pandas    as pd
import numpy     as np
import SimpleITK as sitk

from sklearn.model_selection import train_test_split
from keras_unet.models       import vMonoCTNet
from keras.optimizers        import Adam
from keras.callbacks         import ModelCheckpoint
#from keras.callbacks         import EarlyStopping
from keras_unet.utils        import get_augmented

def MSE(y_true, y_pred):
    return (y_true-y_pred)*(y_true-y_pred)

def MAE(y_true, y_pred):
    return (y_true-y_pred)*100.0

""" Variables """
modelname = 'vMonoCT.h5' 
csvfile   = 'vMonoCT.csv'

dirSave   = '/home/user/vMonoCT/Output'
dirData   = '/home/user/vMonoCT/Dataset'

mhaDataPoly     = os.path.join(dirData,'Train_Poly_Batch.mha');
mhaDataMono     = os.path.join(dirData,'Train_Mono_Batch.mha');

""" Load Data :: Data into shape,dtype, and range (0->1) """
projectionsPoly = sitk.GetArrayFromImage(sitk.ReadImage(mhaDataPoly)    ).astype(dtype=np.float32);
projectionsMono = sitk.GetArrayFromImage(sitk.ReadImage(mhaDataMono)    ).astype(dtype=np.float32);

projectionsPoly = projectionsPoly.reshape(projectionsPoly.shape[0],projectionsPoly.shape[1],projectionsPoly.shape[2],1)
projectionsMono = projectionsMono.reshape(projectionsMono.shape[0],projectionsMono.shape[1],projectionsMono.shape[2],1)

""" Train / Validation split """
projectionsPoly_train, projectionsPoly_test, projectionsMono_train, projectionsMono_test = train_test_split(projectionsPoly, projectionsMono, test_size=0.20, random_state=0,shuffle=True);

""" Train generation with data augmentation """
#https://keras.io/api/preprocessing/image/
train_gen = get_augmented(
    projectionsPoly_train, projectionsMono_train, batch_size=18,
    data_gen_args = dict(
        rotation_range=0.,
        width_shift_range=0.0,
        height_shift_range=0.0,
        shear_range=0,
        zoom_range=0.0,
        horizontal_flip=False,
        vertical_flip=False,
        fill_mode='constant'
    ))

""" Initialize network """
input_shape = projectionsPoly_train[0].shape

network = vMonoCTNet(
    input_shape,
    use_batch_norm=False,
    num_classes=1,
    filters=16,
    dropout=0.0,
    num_layers = 6,
    output_activation='relu'
);

network.summary()

callback_checkpoint = ModelCheckpoint(
    os.path.join(dirSave,modelname),
    verbose = 1,
    monitor = 'val_loss',
    save_best_only=True,
)

#early_stopping = EarlyStopping(
#    monitor   = 'mean_squared_error',
#    min_delta = 1E-5,
#    verbose   = 1,
#    patience  = 25,
#    mode      = 'min',
#)

network.compile(
    optimizer = Adam(learning_rate=1E-5, beta_1=0.9, beta_2=0.999),
    loss      = 'mean_squared_error',
    metrics   = [MSE,MAE]
)

t = time.time();
history = network.fit_generator(
    train_gen,
    steps_per_epoch=np.ceil(projectionsPoly_train.shape[0]/18),
    epochs=300,

    validation_data=(projectionsPoly_test, projectionsMono_test),
    callbacks=[callback_checkpoint] #,early_stopping]
)

print('Elapsed time training: ' + str(round(time.time()-t,4)) + ' seconds')


hist_df = pd.DataFrame(history.history);
with open(csvfile, mode='w') as f:
    hist_df.to_csv(f)

