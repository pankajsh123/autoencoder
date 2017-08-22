from keras.layers import *
from keras.models import *
from scipy import scipy.misc.imresize
from keras.objectives import *
from keras.metrics import binary_crossentropy
import keras.backend as K
import tensorflow as tf

def identity_block(kernel_size, filters):

    def f(input_tensor):
    
        nb_filter1, nb_filter2, nb_filter3 = filters
        
        x = Convolution2D(nb_filter1, 1, 1)(input_tensor)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Convolution2D(nb_filter2, kernel_size, kernel_size,border_mode='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Convolution2D(nb_filter3, 1, 1)(x)
        x = BatchNormalization()(x)

        x = merge([x, input_tensor], mode='sum')
        x = Activation('relu')(x)
        return x
        
    return f

def conv_block(kernel_size, filters,strides=(2, 2)):

    def f(input_tensor):
    
        nb_filter1, nb_filter2, nb_filter3 = filters

        x = Convolution2D(nb_filter1, 1, 1)(input_tensor)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Convolution2D(nb_filter2, kernel_size, strides = strides)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Convolution2D(nb_filter3, 1, 1)(x)
        x = BatchNormalization()(x)

        shortcut = Convolution2D(nb_filter3, 1, 1)(input_tensor)
        shortcut = BatchNormalization()(shortcut)

        x = merge([x, shortcut], mode='sum')
        x = Activation('relu')(x)
        return x
        
    return f

def FCN_Vgg16(input_shape,classes):

    img_input = Input(shape = input_shape)

    # Block 1
    x = Convolution2D(64, 3, strides=(1, 1),padding='same')(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(64, 3,strides=(1, 1),padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 2
    x = Convolution2D(128, 3, strides=(1, 1),padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(128, 3, strides=(1, 1),padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 3
    x = Convolution2D(256, 3, strides=(1, 1),padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(256, 3, strides=(1, 1),padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(256, 3, strides=(1, 1),padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 4
    x = Convolution2D(512, 3, strides=(1, 1),padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(512, 3, strides=(1, 1),padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(512, 3, strides=(1, 1),padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 5
    x = Convolution2D(512, 3, strides=(1, 1),padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(512, 3, strides=(1, 1),padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(512, 3, strides=(1, 1),padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Convolutional layers transfered from fully-connected layers
    x = Convolution2D(4096, 3, strides=(1, 1),padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Convolution2D(4096, 3, strides=(1, 1),padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Convolution2D(4096, 1, strides=(1, 1),padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    
    #classifying layer
    x = Convolution2D(classes, 1, 1, init='he_normal', activation='linear', padding='valid', subsample=(1, 1))(x)

    x = scipy.misc.imresize(x,size=(32, 32),interp='bilinear')

    model = Model(img_input, x)

    return model


def FCN_Resnet50(input_shape,classes):
    
    img_input = Input(shape = input_shape)

    x = Convolution2D(64, 3, 3, subsample=(2, 2), padding='same')(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(64, 3, 3, subsample=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(64, 3, 3, subsample=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(3, [64, 64, 256],strides=(1, 1))(x)
    x = identity_block(3, [64, 64, 256])(x)
    x = identity_block(3, [64, 64, 256])(x)

    x = conv_block(3, [128, 128, 512])(x)
    x = identity_block(3, [128, 128, 512])(x)
    x = identity_block(3, [128, 128, 512])(x)
    x = identity_block(3, [128, 128, 512])(x)

    x = conv_block(3, [256, 256, 1024])(x)
    x = identity_block(3, [256, 256, 1024])(x)
    x = identity_block(3, [256, 256, 1024])(x)
    x = identity_block(3, [256, 256, 1024])(x)
    x = identity_block(3, [256, 256, 1024])(x)
    x = identity_block(3, [256, 256, 1024])(x)

    x = conv_block(3, [512, 512, 2048])(x)
    x = identity_block(3, [512, 512, 2048])(x)
    x = identity_block(3, [512, 512, 2048])(x)
    
    #classifying layer
    x = Convolution2D(classes, 1, 1, init='he_normal', activation='linear', padding='valid', subsample=(1, 1))(x)

    x = scipy.misc.imresize(x,size=(32, 32),interp='bilinear')

    model = Model(img_input, x)

    return model

def loss_(y_true,y_pred,classes):
	
    y_pred = K.reshape(y_pred, (-1,classes))
    log_softmax = tf.nn.log_softmax(y_pred)

    y_true = K.one_hot(tf.to_int32(K.flatten(y_true)), K.int_shape(y_pred)[-1]+1)
    unpacked = tf.unstack(y_true, axis=-1)
    y_true = tf.stack(unpacked[:-1], axis=-1)

    cross_entropy = -K.sum(y_true * log_softmax, axis=1)
    cross_entropy_mean = K.mean(cross_entropy)

	return cross_entropy_mean
	
def accuracy(y_true, y_pred,classes):
	
    y_pred = K.reshape(y_pred, (-1,classes))
    y_true = K.one_hot(tf.to_int32(K.flatten(y_true)),classes + 1)
    unpacked = tf.unstack(y_true, axis=-1)
    legal_labels = ~tf.cast(unpacked[-1], tf.bool)
    y_true = tf.stack(unpacked[:-1], axis=-1)

	return K.sum(tf.to_float(legal_labels & K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)))) / K.sum(tf.to_float		(legal_labels))
	
