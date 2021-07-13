from keras.models import Model
import tensorflow as tf
from keras.layers import * 
from keras import backend as K
from CorrelationLayer import *
from keras.applications import DenseNet121

# This is the master class of the network, in case you build may build many networks
class deepHomo(object):
    # This is the abstract init function
    def __init__(self, image_w, image_h, channel):
        raise NotImplementedError('You have to implement it in the derived class!')

    # This is the normalize function for all class
    def normImage(self, image):
        return (image / 127.5) - 1
    
# The newNetwork class is derived from the master class 'DLNetwork', it will share the normImage function as its function too
    
class JoshuaNetwork(deepHomo):
    # This is the init function for newNetwork class
    
    def __init__(self, image_w, image_h, channel):

        # This is the input in the network
        self.left_input     =   Input(shape=(image_h, image_w,    3))
        self.right_input    =   Input(shape=(image_h, image_w,    3))

        self.model = DenseNet121(input_shape=(image_h,image_w,3), include_top=False)
        self.model.trainable = False
        for l in self.model.layers:
            l.trainable = False
        self.featureLeft    =   self.model(self.left_input)
        self.featureRight   =   self.model(self.right_input)

        # Implement your network architecture here!
        x                   =   Normlized_Cross_Correlation(k=2, d=10, s1=1, s2=1)([self.featureLeft, self.featureRight])
        
        x                   =   Conv2D(128, (5, 5), padding='same', strides=(1,1),   kernel_initializer='glorot_normal')(x)
        x                   =   BatchNormalization()(x)
        x                   =   Activation('relu')(x)

        x                   =   Conv2D(64, (3, 3), padding='same', strides=(1,1),   kernel_initializer='glorot_normal')(x)
        x                   =   BatchNormalization()(x)
        x                   =   Activation('relu')(x)
        
        x                   =   Flatten()(x)
        x                   =   Dropout(0.5)(x)
        x                   =   Dense(1048)(x)

        prediction          =   Dense(2)(x)

        # The model is generated based on the input and output, it is saved in self.model
        self.model = Model(inputs   =   [self.left_input,   self.right_input], outputs  =   prediction)
