# ========================= Classes and Loss Functions ======================= #
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Reshape, Flatten, Dropout, Input, Add, Concatenate, Lambda
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.models import Model
import keras.backend as K
import keras.applications as apps
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import requests
from io import BytesIO
from keras.utils import multi_gpu_model
from tensorflow.python.client import device_lib
import warnings
import pickle
import os.path
warnings.filterwarnings("ignore")
from keras.engine import Layer, InputSpec
from keras import initializers, regularizers, constraints
from keras.utils.generic_utils import get_custom_objects

class InstanceNormalization(Layer):
    """Instance normalization layer (Lei Ba et al, 2016, Ulyanov et al., 2016).
    Normalize the activations of the previous layer at each step,
    i.e. applies a transformation that maintains the mean activation
    close to 0 and the activation standard deviation close to 1.
    # Arguments
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv2D` layer with
            `data_format="channels_first"`,
            set `axis=1` in `InstanceNormalization`.
            Setting `axis=None` will normalize all values in each instance of the batch.
            Axis 0 is the batch dimension. `axis` cannot be set to 0 to avoid errors.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
            When the next layer is linear (also e.g. `nn.relu`),
            this can be disabled since the scaling
            will be done by the next layer.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as input.
    # References
        - [Layer Normalization](https://arxiv.org/abs/1607.06450)
        - [Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/abs/1607.08022)
    """
    def __init__(self,
                 axis=None,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        ndim = len(input_shape)
        if self.axis == 0:
            raise ValueError('Axis cannot be zero')

        if (self.axis is not None) and (ndim == 2):
            raise ValueError('Cannot specify axis for rank 1 tensor')

        self.input_spec = InputSpec(ndim=ndim)

        if self.axis is None:
            shape = (1,)
        else:
            shape = (input_shape[self.axis],)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.built = True

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        reduction_axes = list(range(0, len(input_shape)))

        if (self.axis is not None):
            del reduction_axes[self.axis]

        del reduction_axes[0]

        mean = K.mean(inputs, reduction_axes, keepdims=True)
        stddev = K.std(inputs, reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs - mean) / stddev

        broadcast_shape = [1] * len(input_shape)
        if self.axis is not None:
            broadcast_shape[self.axis] = input_shape[self.axis]

        if self.scale:
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            normed = normed * broadcast_gamma
        if self.center:
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            normed = normed + broadcast_beta
        return normed

    def get_config(self):
        config = {
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(InstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


get_custom_objects().update({'InstanceNormalization': InstanceNormalization})


def res_block(input, filters, kernel_size=(3,3), strides=(1,1), use_dropout=False):
    """
    Instanciate a Keras Resnet Block using sequential API.
    :param input: Input tensor
    :param filters: Number of filters to use
    :param kernel_size: Shape of the kernel for the convolution
    :param strides: Shape of the strides for the convolution
    :param use_dropout: Boolean value to determine the use of dropout
    :return: Keras Model
    """
    #x = ReflectionPadding2D((1,1))(input)
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               padding = 'same',
               strides=strides,)(input)
    x = InstanceNormalization(axis=3)(x)
    x = Activation('relu')(x)

    if use_dropout:
        x = Dropout(0.5)(x)

    #x = ReflectionPadding2D((1,1))(x)
    x = Conv2D(filters=filters,
                kernel_size=kernel_size,
                padding = 'same',
                strides=strides,)(x)
    x = InstanceNormalization(axis=3)(x)

    # Two convolution layers followed by a direct connection between input and output
    merged = Add()([input, x])
    return merged





# blur loss == Fourier loss
def blur_loss(y_true, y_pred):

    y_true = K.abs(tf.spectral.rfft2d(y_true))
    y_pred = K.abs(tf.spectral.rfft2d(y_pred))

    return 0.5 * K.mean(K.square(y_true - y_pred))


def perceptual_MSE_loss(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred))


def perceptual_VGG16_loss(y_true, y_pred):
    vgg = apps.VGG16(include_top=False, weights='imagenet', input_shape=(new_image_shape[0], new_image_shape[1], 3))
    #vgg = apps.VGG16(include_top=False, weights='imagenet', input_shape=(image_shape[0]*2, image_shape[1]*2, 3))
    loss_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
    loss_model.trainable = False

    y_true3d = K.concatenate([y_true,y_true,y_true])
    y_pred3d = K.concatenate([y_pred,y_pred,y_pred])
    loss = K.mean(K.square(loss_model(y_true3d) - loss_model(y_pred3d)))
    return loss



# ======================= GAN models ======================== #
def Generator():
    """Build generator architecture."""
    # Current version : ResNet block
    inputs = Input(shape=new_image_shape)

    x = Conv2D(filters=64, kernel_size=7, strides=1, padding='same')(inputs)
    x = InstanceNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=128, kernel_size=3, strides=2, padding='same')(x)
    x = InstanceNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=256, kernel_size=3, strides=2, padding='same')(x)
    x = InstanceNormalization(axis=3)(x)
    x = Activation('relu')(x)

    # Apply 9 ResNet blocks
    for i in range(9):
        x = res_block(x, 256, use_dropout=True)

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(x)
    x = InstanceNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(x)
    x = InstanceNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=1, kernel_size=7, padding='same')(x)
    x = Activation('tanh')(x)

    # Add direct connection from input to output
    outputs = Add()([x, inputs])
    outputs = Lambda(lambda z: z/2)(outputs)

    model = Model(inputs=inputs, outputs=outputs, name='Generator')
    if (num_gpu>1):
        model = multi_gpu_model(model, gpus=num_gpu)

    return model


def Discriminator():

    model = Sequential()
    model.add(Conv2D(64, kernel_size=(5, 5), strides=(2, 2), padding='same', input_shape=new_image_shape))
    model.add(LeakyReLU(0.2))
    model.add(Conv2D(128, kernel_size=(5, 5), strides=(2, 2)))
    model.add(LeakyReLU(0.2))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    d_opt = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=d_opt, loss='binary_crossentropy')
    if (num_gpu>1):
        model = multi_gpu_model(model, gpus=num_gpu)

    return model


def GAN(generator, discriminator):

    discriminator.trainable = False

    inputs = Input(shape=new_image_shape)
    generated_images = generator(inputs)
    outputs = discriminator(generated_images)

    model = Model(inputs=inputs, outputs=[generated_images, outputs])

    gan_opt = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    loss = [perceptual_VGG16_loss, 'binary_crossentropy']
    loss_weights = [1, 0.1]

    model.compile(optimizer=gan_opt, loss=loss, loss_weights=loss_weights)

    return model
