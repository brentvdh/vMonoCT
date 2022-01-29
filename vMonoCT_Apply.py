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
import numpy     as np
import SimpleITK as sitk

from keras_unet.models import vMonoCTNet

""" Variables """
modelname = 'vMonoCT.h5'

dirSave   = '/home/user/vMonoCT/Output'
dirData   = '/home/user/vMonoCT/Dataset'

mhaPoly   = os.path.join(dirData,'Raw_AI_input.mha');
mhaOutput = os.path.join(dirSave,'Raw_AI_output.mha');

""" Load Data :: Data into shape,dtype, and range (0->1) """
projectionsVal = sitk.GetArrayFromImage(sitk.ReadImage(mhaPoly)).astype(dtype=np.float32);
projectionsVal = projectionsVal.reshape(projectionsVal.shape[0],projectionsVal.shape[1],projectionsVal.shape[2],1)

#https://keras.io/api/preprocessing/image/
""" Initialize network """
network = vMonoCTNet(
    projectionsVal[0].shape,
    use_batch_norm=False,
    num_classes=1,
    filters=16,
    dropout=0.0,
    num_layers = 6,
    output_activation='relu'
);

network.load_weights(os.path.join(dirSave,modelname))

t = time.time()
PRED = network.predict(projectionsVal);
writer = sitk.ImageFileWriter()
writer.SetFileName(mhaOutput)
writer.Execute( sitk.GetImageFromArray(PRED))

print('Elapsed time testing: ' + str(round(time.time()-t,4)) + ' seconds')
