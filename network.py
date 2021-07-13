# You can call the needed layers only
# Please call the packages that are required to define the network architecture
from keras.models import Model
import tensorflow as tf
from keras.layers import * 
from keras import backend as K
from keras.applications import DenseNet121

class NCC(Layer):
    def __init__(self, **kwargs):
        super(NCC, self).__init__(**kwargs)

    def build(self, input_shape):
        super(NCC, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        feature_1, feature_2 = x
        feature_shape = feature_1._keras_shape

        print(feature_shape)
        print(feature_2._keras_shape)
        assert feature_shape == feature_1._keras_shape

        self.H = feature_shape[1]
        self.W = feature_shape[2]
        self.C = feature_shape[3]

        fea1 = K.reshape(feature_1,shape=(-1,self.H*self.W, self.C))
        fea2 = K.permute_dimensions(K.reshape(feature_2,shape=(-1,self.H*self.W, self.C)),(0, 2, 1))

        out = K.batch_dot(fea1,fea2)
        out = K.reshape(out, shape=(-1, self.H, self.W, self.H*self.W))
        
        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], self.H, self.W, self.H*self.W)

    

if K.backend() == 'tensorflow':
    import tensorflow as tf

    def K_meshgrid(x, y):
        return tf.meshgrid(x, y)

    def K_linspace(start, stop, num):
        return tf.linspace(start, stop, num)

else:
    raise Exception("Only 'tensorflow' is supported as backend")


class BilinearInterpolation(Layer):
    """Performs bilinear interpolation as a keras layer
    References
    ----------
    [1]  Spatial Transformer Networks, Max Jaderberg, et al.
    [2]  https://github.com/skaae/transformer_network
    [3]  https://github.com/EderSantana/seya
    """

    def __init__(self, output_size, **kwargs):
        self.output_size = output_size
        super(BilinearInterpolation, self).__init__(**kwargs)

    def compute_output_shape(self, input_shapes):
        height, width = self.output_size
        num_channels = input_shapes[0][-1]
        return (None, height, width, num_channels)

    def call(self, tensors, mask=None):
        X, transformation = tensors
        output = self._transform(X, transformation, self.output_size)
        return output

    def _interpolate(self, image, sampled_grids, output_size):

        batch_size = K.shape(image)[0]
        height = K.shape(image)[1]
        width = K.shape(image)[2]
        num_channels = K.shape(image)[3]

        x = K.cast(K.flatten(sampled_grids[:, 0:1, :]), dtype='float32')
        y = K.cast(K.flatten(sampled_grids[:, 1:2, :]), dtype='float32')


        x = .5 * (x + 1.0) * K.cast(width, dtype='float32')
        y = .5 * (y + 1.0) * K.cast(height, dtype='float32')

        x0 = K.cast(x, 'int32')
        x1 = x0 + 1
        y0 = K.cast(y, 'int32')
        y1 = y0 + 1

        max_x = int(K.int_shape(image)[2] - 1)
        max_y = int(K.int_shape(image)[1] - 1)

        x0 = K.clip(x0, 0, max_x)
        x1 = K.clip(x1, 0, max_x)
        y0 = K.clip(y0, 0, max_y)
        y1 = K.clip(y1, 0, max_y)

        pixels_batch = K.arange(0, batch_size) * (height * width)
        pixels_batch = K.expand_dims(pixels_batch, axis=-1)
        flat_output_size = output_size[0] * output_size[1]
        base = K.repeat_elements(pixels_batch, flat_output_size, axis=1)
        base = K.flatten(base)

        base_y0 = y0 * width
        base_y0 = base + base_y0
        base_y1 = y1 * width
        base_y1 = base_y1 + base

        indices_a = base_y0 + x0
        indices_b = base_y1 + x0
        indices_c = base_y0 + x1
        indices_d = base_y1 + x1

        flat_image = K.reshape(image, shape=(-1, num_channels))
        flat_image = K.cast(flat_image, dtype='float32')
        pixel_values_a = K.gather(flat_image, indices_a)
        pixel_values_b = K.gather(flat_image, indices_b)
        pixel_values_c = K.gather(flat_image, indices_c)
        pixel_values_d = K.gather(flat_image, indices_d)

        x0 = K.cast(x0, 'float32')
        x1 = K.cast(x1, 'float32')
        y0 = K.cast(y0, 'float32')
        y1 = K.cast(y1, 'float32')

        area_a = K.expand_dims(((x1 - x) * (y1 - y)), 1)
        area_b = K.expand_dims(((x1 - x) * (y - y0)), 1)
        area_c = K.expand_dims(((x - x0) * (y1 - y)), 1)
        area_d = K.expand_dims(((x - x0) * (y - y0)), 1)

        values_a = area_a * pixel_values_a
        values_b = area_b * pixel_values_b
        values_c = area_c * pixel_values_c
        values_d = area_d * pixel_values_d
        return K.clip(values_a + values_b + values_c + values_d, -1., 1.)

    def _make_regular_grids(self, batch_size, height, width):
        # making a single regular grid
        x_linspace = K_linspace(-1., 1., width)
        y_linspace = K_linspace(-1., 1., height)
        x_coordinates, y_coordinates = K_meshgrid(x_linspace, y_linspace)
        x_coordinates = K.flatten(x_coordinates)
        y_coordinates = K.flatten(y_coordinates)
        ones = K.ones_like(x_coordinates)
        grid = K.concatenate([x_coordinates, y_coordinates, ones], 0)

        grid = K.flatten(grid)
        grids = K.tile(grid, K.stack([batch_size]))
        return K.reshape(grids, (batch_size, 3, height * width))

    def _transform(self, X, homography_transformation, output_size):
        batch_size, num_channels = K.shape(X)[0], K.shape(X)[3]
        transformations = K.reshape(homography_transformation,
                                    shape=(batch_size, 3, 3))
        transformations = K.cast(transformations, 'float32')
        regular_grids = self._make_regular_grids(batch_size, output_size[0], output_size[1])
        
        sampled_grids = tf.matmul(transformations, regular_grids)
        interpolated_image = self._interpolate(X, sampled_grids, output_size)
        new_shape = (batch_size, output_size[0], output_size[1], 3)
        interpolated_image = K.reshape(interpolated_image, new_shape)

        return interpolated_image
    

class HomographyWarp(Layer):
    """Performs bilinear interpolation as a keras layer
    References
    ----------
    [1]  Spatial Transformer Networks, Max Jaderberg, et al.
    [2]  https://github.com/skaae/transformer_network
    [3]  https://github.com/EderSantana/seya
    """

    def __init__(self, output_size, **kwargs):
        self.output_size = output_size
        super(HomographyWarp, self).__init__(**kwargs)

    def compute_output_shape(self, input_shapes):
        height, width = self.output_size
        return (None, height, width)

    def call(self, tensors, mask=None):
        transformation = tensors
        output = self._transform(transformation)
        return output

    def _make_regular_grids(self, batch_size, height, width):
        # making a single regular grid
        x_linspace = K_linspace(-1., 1., width)
        y_linspace = K_linspace(-1., 1., height)
        x_coordinates, y_coordinates = K_meshgrid(x_linspace, y_linspace)
        x_coordinates = K.flatten(x_coordinates)
        y_coordinates = K.flatten(y_coordinates)
        ones = K.ones_like(x_coordinates)
        grid = K.concatenate([x_coordinates, y_coordinates, ones], 0)

        grid = K.flatten(grid)
        grids = K.tile(grid, K.stack([batch_size]))
        return K.reshape(grids, (batch_size, 3, height * width))

    def _transform(self, homography_transformation):
        batch_size = K.shape(homography_transformation)[0]
        
        ones = K.ones((batch_size,1))
        transformations = K.reshape(homography_transformation,
                                    shape=(batch_size, 3, 3))
        
        transformations = K.cast(transformations, 'float32')
        regular_grids = self._make_regular_grids(batch_size, self.output_size[0], self.output_size[1])
        
        sampled_grids = tf.matmul(transformations, regular_grids)

        return sampled_grids[:,0:2,:] / K.repeat(sampled_grids[:,2,:],2)


class AffineWarp(Layer):
    """Performs bilinear interpolation as a keras layer
    References
    ----------
    [1]  Spatial Transformer Networks, Max Jaderberg, et al.
    [2]  https://github.com/skaae/transformer_network
    [3]  https://github.com/EderSantana/seya
    """

    def __init__(self, output_size, **kwargs):
        self.output_size = output_size
        super(AffineWarp, self).__init__(**kwargs)

    def compute_output_shape(self, input_shapes):
        height, width = self.output_size
        return (None, height, width)

    def call(self, tensors, mask=None):
        transformation = tensors
        output = self._transform(transformation)
        return output

    def _make_regular_grids(self, batch_size, height, width):
        # making a single regular grid
        x_linspace = K_linspace(-1., 1., width)
        y_linspace = K_linspace(-1., 1., height)
        x_coordinates, y_coordinates = K_meshgrid(x_linspace, y_linspace)
        x_coordinates = K.flatten(x_coordinates)
        y_coordinates = K.flatten(y_coordinates)
        ones = K.ones_like(x_coordinates)
        grid = K.concatenate([x_coordinates, y_coordinates, ones], 0)

        grid = K.flatten(grid)
        grids = K.tile(grid, K.stack([batch_size]))
        return K.reshape(grids, (batch_size, 3, height * width))

    def _transform(self, homography_transformation):
        batch_size = K.shape(homography_transformation)[0]
        
        ones = K.ones((batch_size,1))
        transformations = K.reshape(homography_transformation,
                                    shape=(batch_size, 2, 3))
        
        transformations = K.cast(transformations, 'float32')
        regular_grids = self._make_regular_grids(batch_size, self.output_size[0], self.output_size[1])
        
        sampled_grids = tf.matmul(transformations, regular_grids)

        return sampled_grids

def get_initial_weights(output_size):
    b = np.zeros((2,3), dtype='float32')
    b[0,0] = 1
    b[0,2] = 160
    b[1,1] = 1
    #b[2,2] = 1
    W = np.zeros((output_size, 6), dtype='float32')
    weights = [W, b.flatten()]
    return weights

def get_initialHomo_weights(output_size):
    b = np.zeros((3,3), dtype='float32')
    b[0,0] = 1
    b[0,2] = 160
    b[1,1] = 1
    b[2,2] = 1
    W = np.zeros((output_size, 9), dtype='float32')
    weights = [W, b.flatten()]
    return weights

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
        left_input = Input(shape=(image_h,image_w,3))
        right_input = Input(shape=(image_h,image_w,3))
        
        densenet = DenseNet121(weights='imagenet', include_top=False)
        
        featureEx = Model(inputs=densenet.input,outputs=densenet.get_layer('pool4_pool').output)
        
        featureEx.trainable = False
        for l in featureEx.layers:
            l.trainable = False

        featureLeft    =   featureEx(left_input)
        featureRight   =   featureEx(right_input)

        corr = NCC()([featureLeft,featureRight])

        x =   Conv2D(128, (7, 7),padding='same', strides=(1,1),   kernel_initializer='glorot_normal')(corr)
        x =   BatchNormalization()(x)
        x =   Activation('relu')(x)

        x =   Conv2D(64, (5, 5), padding='same', strides=(1,1),   kernel_initializer='glorot_normal')(x)
        x =   BatchNormalization()(x)
        x =   Activation('relu')(x)
                
        x =   Conv2D(32, (3, 3), padding='same', strides=(1,1),   kernel_initializer='glorot_normal')(x)
        x =   BatchNormalization()(x)
        x =   Activation('relu')(x)

        x =   Conv2D(16, (1, 1), padding='same', strides=(1,1),   kernel_initializer='glorot_normal')(x)
        x =   BatchNormalization()(x)
        x =   Activation('relu')(x)

        x =   Flatten()(x)

        x =   Dense(64)(x)
        
        weights = get_initial_weights(64)
        prediction =   Dense(6, weights=weights)(x)

        # The model is generated based on the input and output, it is saved in self.model
        self.model = Model([left_input,right_input],prediction)

# The newNetwork class is derived from the master class 'DLNetwork', it will share the normImage function as its function too
class PaoNetwork(deepHomo):
    # This is the init function for newNetwork class
    def __init__(self, image_w, image_h, channel):
        
        # This is the input in the network
        left_input = Input(shape=(image_h,image_w,3))
        right_input = Input(shape=(image_h,image_w,3))
        
        densenet = DenseNet121(weights='imagenet', include_top=False)
        
        featureEx = Model(inputs=densenet.input,outputs=densenet.get_layer('pool4_pool').output)
        
        featureEx.trainable = False
        for l in featureEx.layers:
            l.trainable = False

        featureLeft    =   featureEx(left_input)
        featureRight   =   featureEx(right_input)

        corr = NCC()([featureLeft,featureRight])

        x =   Conv2D(128, (7, 7),padding='same', strides=(1,1),   kernel_initializer='glorot_normal')(corr)
        x =   BatchNormalization()(x)
        x =   Activation('relu')(x)

        x =   Conv2D(64, (5, 5), padding='same', strides=(1,1),   kernel_initializer='glorot_normal')(x)
        x =   BatchNormalization()(x)
        x =   Activation('relu')(x)
                
        x =   Conv2D(32, (3, 3), padding='same', strides=(1,1),   kernel_initializer='glorot_normal')(x)
        x =   BatchNormalization()(x)
        x =   Activation('relu')(x)

        x =   Conv2D(16, (1, 1), padding='same', strides=(1,1),   kernel_initializer='glorot_normal')(x)
        x =   BatchNormalization()(x)
        x =   Activation('relu')(x)

        x =   Flatten()(x)

        x =   Dense(64)(x)
        
        weights = get_initialHomo_weights(64)
        prediction =   Dense(8, weights=weights)(x)

        # The model is generated based on the input and output, it is saved in self.model
        self.model = Model([left_input,right_input],prediction)


class Deep_homography(deepHomo):
    def __init__(self, image_w, image_h, channel):

        left_input = Input(shape=(image_h,image_w,3))
        right_input = Input(shape=(image_h,image_w,3))
        
        input = Concatenate()([left_input,right_input])
        
        x = Conv2D(64, (3, 3), padding='same', strides=(1,1))(input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(64, (3, 3), padding='same', strides=(1,1))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2,2))(x)

        x = Conv2D(64, (3, 3), padding='same', strides=(1,1))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(64, (3, 3), padding='same', strides=(1,1))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2,2))(x)

        x = Conv2D(128, (3, 3), padding='same', strides=(1,1))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(128, (3, 3), padding='same', strides=(1,1))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2,2))(x)

        x = Conv2D(128, (3, 3), padding='same', strides=(1,1))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(128, (3, 3), padding='same', strides=(1,1))(x)
        x = BatchNormalization()(x)

        x = Flatten()(x)
        x = Dropout(0.5)(x)
        x = Dense(1024)(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)
        prediction = Dense(8)(x)
        
        self.model = Model([left_input,right_input], prediction)


# The newNetwork class is derived from the master class 'DLNetwork', it will share the normImage function as its function too
class NewNetwork(deepHomo):
    # This is the init function for newNetwork class
    def __init__(self, image_w, image_h, channel):
        
        # This is the input in the network
        left_input = Input(shape=(image_h,image_w,3))
        right_input = Input(shape=(image_h,image_w,3))
        
        densenet = DenseNet121(weights='imagenet', include_top=False)
        
        featureEx = Model(inputs=densenet.input,outputs=densenet.get_layer('pool4_pool').output)
        
        featureEx.trainable = False
        for l in featureEx.layers:
            l.trainable = False

        featureLeft    =   featureEx(left_input)
        featureRight   =   featureEx(right_input)

        corr = NCC()([featureLeft,featureRight])

        x =   Conv2D(128, (7, 7),padding='same', strides=(1,1),   kernel_initializer='glorot_normal')(corr)
        x =   BatchNormalization()(x)
        x =   Activation('relu')(x)

        x =   Conv2D(64, (5, 5), padding='same', strides=(1,1),   kernel_initializer='glorot_normal')(x)
        x =   BatchNormalization()(x)
        x =   Activation('relu')(x)
                
        x =   Conv2D(32, (3, 3), padding='same', strides=(1,1),   kernel_initializer='glorot_normal')(x)
        x =   BatchNormalization()(x)
        x =   Activation('relu')(x)

        x =   Conv2D(16, (1, 1), padding='same', strides=(1,1),   kernel_initializer='glorot_normal')(x)
        x =   BatchNormalization()(x)
        x =   Activation('relu')(x)

        x =   Flatten()(x)

        x =   Dense(64)(x)
        
        prediction =   Dense(8)(x)

        # The model is generated based on the input and output, it is saved in self.model
        self.model = Model([left_input,right_input],prediction)