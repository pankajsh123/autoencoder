import numpy as np
import matplotlib.pyplot as plt
import os
from keras.layers import *
from keras.models import *
from scipy import scipy.misc.imresize
from keras.metrics import binary_accuracy
from keras.models import load_model
from utils.loss_function import *
from utils.metrics import *

if vgg:
	model = FCN_Vgg16(input_shape,classes)

else resnet:
	model = FCN_Resnet50(input_shape,classes)

loss_fnc = loss_
acc = [accuracy]

model.compile(optimizer='sgd', loss = loss_fnc,metrics = acc)
model.fit(trainY,trainY,epochs=1,batch_size=16,shuffle=True)

