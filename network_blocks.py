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



class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""
    def _merge_function(self, inputs):
        alpha = K.random_uniform((batch_size, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])



# cGAN-related functions

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





# CIN-GAN-related functions

vgg19 = apps.VGG19(include_top=False, weights='imagenet', input_shape=(new_height, new_width, 3))

def content_VGG19_loss(y_true, y_pred):
    loss_model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv2').output)
    loss_model.trainable = False

    #y_true = preprocess(y_true)
    #y_pred = preprocess(y_pred)

    y_true3d = (K.concatenate([y_true,y_true,y_true])+1)*255
    y_pred3d = (K.concatenate([y_pred,y_pred,y_pred])+1)*255
    loss = K.mean(K.square(loss_model(y_true3d) - loss_model(y_pred3d)))
    return loss


# source:  https://github.com/keras-team/keras/blob/master/examples/neural_style_transfer.py

def gram_matrix(x):
    assert K.ndim(x) == 3
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram


def style_features_loss(style, gen):
    result = 0
    for i in range(batch_size):
        S = gram_matrix(style[i])
        G = gram_matrix(gen[i])
        channels = 3
        size = new_height * new_width #img_h * img_w
        result +=  K.sum(K.abs(S - G)) / (4. * (channels ** 2) * (size ** 2))

    return result / batch_size

def style_VGG19_loss(y_true, y_pred):

    feature_layer_names = ['block1_conv1', 'block2_conv1' 'block3_conv1', 'block4_conv1']

    y_true3d = K.concatenate([y_true,y_true,y_true])
    y_pred3d = K.concatenate([y_pred,y_pred,y_pred])

    loss_model1 = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block1_conv1').output)
    loss_model1.trainable = False
    style_features1 = loss_model1(y_true3d)
    gen_img_features1 = loss_model1(y_pred3d)
    s1 = style_features_loss(style_features1, gen_img_features1)
    loss1 = s1

    loss_model2 = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block2_conv1').output)
    loss_model2.trainable = False
    style_features2 = loss_model2(y_true3d)
    gen_img_features2 = loss_model2(y_pred3d)
    s2 = style_features_loss(style_features2, gen_img_features2)
    loss2 = s2

    loss_model3 = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block3_conv1').output)
    loss_model3.trainable = False
    style_features3 = loss_model3(y_true3d)
    gen_img_features3 = loss_model3(y_pred3d)
    s3 = style_features_loss(style_features3, gen_img_features3)
    loss3 = s3

    loss_model4 = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block4_conv1').output)
    loss_model4.trainable = False
    style_features4 = loss_model4(y_true3d)
    gen_img_features4 = loss_model4(y_pred3d)
    s4 = style_features_loss(style_features4, gen_img_features4)
    loss4 = s4

#    loss_model5 = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv1').output)
#    loss_model5.trainable = False
#    style_features5 = loss_model5(y_true)
#    gen_img_features5 = loss_model5(y_pred)
#    s5 = style_features_loss(style_features5, gen_img_features5)
#    loss5 = s5 / len(feature_layer_names)

    loss = loss1 + loss2 + loss3 + loss4

    return loss / len(feature_layer_names)



def perceptual_VGG16_loss(y_true, y_pred):
    vgg = apps.VGG16(include_top=False, weights='imagenet', input_shape=(new_image_shape[0], new_image_shape[1], 3))
    loss_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
    loss_model.trainable = False

    y_true3d = K.concatenate([y_true,y_true,y_true])
    y_pred3d = K.concatenate([y_pred,y_pred,y_pred])
    loss = K.mean(K.abs(loss_model(y_true3d) - loss_model(y_pred3d)))
    return loss

def blur_loss(y_true, y_pred):

    y_true = K.abs(tf.spectral.rfft2d(y_true))
    y_pred = K.abs(tf.spectral.rfft2d(y_pred))

    return 0.5 * K.mean(K.square(y_true - y_pred))


def gradient_penalty_loss(y_true, y_pred, averaged_samples):
    """
    Computes gradient penalty based on prediction and weighted real / fake samples
    """
    gradients = K.gradients(y_pred, averaged_samples)[0]
    # compute the euclidean norm by squaring ...
    gradients_sqr = K.square(gradients)
    #   ... summing over the rows ...
    gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    #   ... and sqrt
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    # compute lambda * (1 - ||grad||)^2 still for each single sample
    gradient_penalty = K.square(1 - gradient_l2_norm)
    # return the mean as loss over all the batch samples
    return K.mean(gradient_penalty)


def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)



c1 = K.variable([0])
c2 = K.variable([0])
c3 = K.variable([0])

def CIN_res_block(input, filters, kernel_size=(3,3), strides=(1,1), use_dropout=False):
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
    x_cond1 = InstanceNormalization()(x)
    x_cond2 = InstanceNormalization()(x)
    x_cond3 = InstanceNormalization()(x)
    x_cond1 = Lambda(lambda x: x * c1)(x_cond1)
    x_cond2 = Lambda(lambda x: x * c2)(x_cond2)
    x_cond3 = Lambda(lambda x: x * c3)(x_cond3)
    x = Add()([x_cond1, x_cond2, x_cond3])
    x = Activation('relu')(x)

    if use_dropout:
        x = Dropout(0.5)(x)

    #x = ReflectionPadding2D((1,1))(x)
    x = Conv2D(filters=filters,
                kernel_size=kernel_size,
                padding = 'same',
                strides=strides,)(x)
    x_cond1 = InstanceNormalization()(x)
    x_cond2 = InstanceNormalization()(x)
    x_cond3 = InstanceNormalization()(x)
    x_cond1 = Lambda(lambda x: x * c1)(x_cond1)
    x_cond2 = Lambda(lambda x: x * c2)(x_cond2)
    x_cond3 = Lambda(lambda x: x * c3)(x_cond3)
    x = Add()([x_cond1, x_cond2, x_cond3])
    # Two convolution layers followed by a direct connection between input and output
    merged = Add()([input, x])
    return merged



def build_CIN_generator():

    inputs = Input(shape=new_image_shape)

    x = Conv2D(filters=64, kernel_size=7, strides=1, padding='same')(inputs)
    x_cond1 = InstanceNormalization(name="BN1_1")(x)
    x_cond2 = InstanceNormalization(name="BN1_2")(x)
    x_cond3 = InstanceNormalization(name="BN1_3")(x)

    x_cond1 = Lambda(lambda x: x * c1)(x_cond1) # will multiply by 0 if not 1st condition
    x_cond2 = Lambda(lambda x: x * c2)(x_cond2)
    x_cond3 = Lambda(lambda x: x * c3)(x_cond3)
    x = Add()([x_cond1, x_cond2, x_cond3])
    x = Activation('relu')(x)

    x = Conv2D(filters=128, kernel_size=3, strides=2, padding='same')(x)
    x_cond1 = InstanceNormalization(name="BN2_1")(x)
    x_cond2 = InstanceNormalization(name="BN2_2")(x)
    x_cond3 = InstanceNormalization(name="BN2_3")(x)

    x_cond1 = Lambda(lambda x: x * c1)(x_cond1) # will multiply by 0 if not 1st condition
    x_cond2 = Lambda(lambda x: x * c2)(x_cond2)
    x_cond3 = Lambda(lambda x: x * c3)(x_cond3)
    x = Add()([x_cond1, x_cond2, x_cond3])
    x = Activation('relu')(x)

    x = Conv2D(filters=256, kernel_size=3, strides=2, padding='same')(x)
    x_cond1 = InstanceNormalization(name="BN3_1")(x)
    x_cond2 = InstanceNormalization(name="BN3_2")(x)
    x_cond3 = InstanceNormalization(name="BN3_3")(x)

    x_cond1 = Lambda(lambda x: x * c1)(x_cond1) # will multiply by 0 if not 1st condition
    x_cond2 = Lambda(lambda x: x * c2)(x_cond2)
    x_cond3 = Lambda(lambda x: x * c3)(x_cond3)
    x = Add()([x_cond1, x_cond2, x_cond3])
    x = Activation('relu')(x)

    # Apply 9 ResNet blocks
    for i in range(12):
        x = res_block(x, 256, use_dropout=True)

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(x)
    #x = InstanceNormalization()(x)
    x = Activation('relu')(x)

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(x)
    #x = InstanceNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=1, kernel_size=7, padding='same')(x)
    x = Activation('tanh')(x)

    # Add direct connection from input to output
    outputs = Add()([x, inputs])
    outputs = Lambda(lambda z: z/2)(outputs)

    model = Model(inputs=inputs, outputs=outputs)

    return model



def build_CIN_critic():

    model = Sequential()

    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same", input_shape=new_image_shape))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Conv2D(512, kernel_size=3, strides=1, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Conv2D(8, kernel_size=3, strides=1, padding="same"))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Dense(1))

    img = Input(shape=new_image_shape)
    validity = model(img)

    return Model(img, validity)



#-------------------------------
# Construct Computational Graph
#       for the Critic
#-------------------------------
def CIN_Critic_model(generator, critic):
    # Freeze generator's layers while training critic
    generator.trainable = False

    # Image input (real sample)
    real_img = Input(shape=img_shape)

    # Noise input
    z_disc = Input(shape=img_shape)

    # Generate image based of noise (fake sample)
    fake_img = generator(z_disc)

    # Discriminator determines validity of the real and fake images
    fake = critic(fake_img)
    valid = critic(real_img)

    # Construct weighted average between real and fake images
    interpolated_img = RandomWeightedAverage()([real_img, fake_img])

    # Determine validity of weighted sample
    validity_interpolated = critic(interpolated_img)

    # Use Python partial to provide loss function with additional
    # 'averaged_samples' argument
    partial_gp_loss = partial(gradient_penalty_loss,
                      averaged_samples=interpolated_img)
    partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names

    critic_model = Model(inputs=[real_img, z_disc],
                        outputs=[valid, fake, validity_interpolated])

    critic_model.compile(loss=[wasserstein_loss,
                               wasserstein_loss,
                               partial_gp_loss],
                        optimizer=critic_opt,
                        loss_weights=[1, 1, 10])

    #print(critic_model.summary())
    return critic_model

#-------------------------------
# Construct Computational Graph
#         for Generator
#-------------------------------
def CIN_Generator_model(generator, critic):
    # For the generator we freeze the critic's layers
    critic.trainable = False
    generator.trainable = True

    # Sampled noise for input to generator
    z_gen = Input(shape=img_shape)

    # Generate images based of noise
    img = generator(z_gen)

    # Discriminator determines validity
    valid = critic(img)

    # Defines generator model
    generator_model = Model(inputs=z_gen, outputs=[img, img, valid])
    generator_model.compile(loss=[perceptual_VGG16_loss, style_VGG19_loss, wasserstein_loss],
                            optimizer=gen_opt,
                            loss_weights=[1, 1, 0.1])

    # print(generator_model.summary())

    return generator_model

