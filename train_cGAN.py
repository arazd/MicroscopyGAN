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
import shutil, sys
import dataset_utils
import metrics
import network_blocks

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('image_limit', -1, """Limit .""")
tf.app.flags.DEFINE_integer('batch_size', 100, """Batch size.""")
tf.app.flags.DEFINE_integer('num_gpu', 2, """Number of available GPUs (default = 2).""")
tf.app.flags.DEFINE_integer('num_iters', 3000, """Number of parameter updates.""")
tf.app.flags.DEFINE_string('dir', 'None', """Directory to save results.""")
images_limit = FLAGS.images_limit
batch_size = FLAGS.batch_size
num_gpu = FLAGS.num_gpu
dir = FLAGS.dir

height = 128
width = 128
image_shape = (height, width, 1)

# TRAINING
num_iterations = FLAGS.num_iters
epochs = int(num_iterations*batch_size/images_limit)
critic_updates = 1

num_batches = int(X_train.shape[0] / batch_size)



# ======================= Load Dataset and Normalize ======================== #

X1 = np.load('Drosophila_X.npy').reshape(-1, height, width, 1)
Y1 = np.load('Drosophila_Y.npy').reshape(-1, height, width, 1)

X2 = np.load('Retina_X.npy').reshape(-1, height, width, 1)
Y2 = np.load('Retina_Y.npy').reshape(-1, height, width, 1)

X3 = np.load('Synthetic_tubulin_tubules_X').reshape(-1, height, width, 1)
Y3 = np.load('Synthetic_tubulin_tubules_Y').reshape(-1, height, width, 1)


X1, Y1, X_eval_1, Y_eval_1 = dataset_utils.normalize_split(X1, Y1, images_limit, eval_limit=100)
print("Loaded X1")

X2, Y2, X_eval_2, Y_eval_2 = dataset_utils.normalize_split(X2, Y2, images_limit, eval_limit=100)
print("Loaded X2")

X3, Y3, X_eval_3, Y_eval_3 = dataset_utils.normalize_split(X3, Y3, images_limit, eval_limit=100)
print("Loaded X3")



# combine the training data
X_train = np.concatenate((X1, X2, X3))
Y_train = np.concatenate((Y1, Y2, Y3))

X_train, Y_train = dataset_utils.unison_shuffled_copies(X_train, Y_train)

print('Max: ', np.max(X_train))
print('Min: ', np.min(X_train))
print('X shape: ', X_train.shape)
print('Y shape: ', Y_train.shape)

new_height = X1.shape[1]
new_width = X1.shape[2]
new_image_shape = (new_height, new_width, 1)



# creating a directory to save the results
if (dir!='None')
    os.chdir(dir)

if (os.path.isdir("Microscopy_cGAN_results")==False):
    os.mkdir("Microscopy_cGAN_results")
os.chdir("Microscopy_cGAN_results")





# create models and compile them

# Generator model
generator = network_blocks.Generator()
#print(generator.summary())

# Discriminator model
discriminator = network_blocks.Discriminator()
#print(discriminator.summary())

# complete the full GAN
gan = network_blocks.GAN(generator, discriminator)



pbar = tqdm(total=epochs * num_batches)

D_loss_test1 = []
G_loss_test1 = []
ssim_scores1 = []
psnr_scores1 = []

D_loss_test2 = []
G_loss_test2 = []
ssim_scores2 = []
psnr_scores2 = []

D_loss_test3 = []
G_loss_test3 = []
ssim_scores3 = []
psnr_scores3 = []

## set the weights based on previously trained model
#get_custom_objects().update({'InstanceNormalization': InstanceNormalization})
#
#discriminator_p = load_model('discriminator-20.h5')
#generator_p = load_model('generator-20.h5')
#
#discriminator.set_weights(discriminator_p.get_weights())
#generator.set_weights(generator_p.get_weights())
#
#print("======================================================================")
#print('Loaded existing models!')
#print("======================================================================")

# ======================= Training ======================== #

for epoch in range(1, epochs+1):

    X_train, Y_train = unison_shuffled_copies(X_train, Y_train)

    for index in range(num_batches):

        pbar.update(1)

        # Generative data
        training_data = Y_train[index * batch_size: (index + 1) * batch_size]
        blurred_batch = X_train[index * batch_size: (index + 1) * batch_size]

        rotation_param = np.random.choice(5) # augmenting data by rotation
        for _ in range(rotation_param):
            training_data = np.rot90(training_data, axes=(1,2))
            blurred_batch = np.rot90(blurred_batch, axes=(1,2))

        generated_data = generator.predict_on_batch(blurred_batch)

        # Training data chosen from Mnist samples


        for _ in range(critic_updates):
            X = np.vstack((generated_data, training_data))
            y = np.ones(2 * batch_size)
            y[:batch_size] = 0

            # Train discriminator
            d_loss = discriminator.train_on_batch(x=X, y=y)


        # Train generator (Seemingly train GAN but the discriminator in the model is disabled to train.)
        y = np.ones(batch_size)
        g_loss = gan.train_on_batch(x=blurred_batch, y=[training_data, y])


        for X_ev, Y_ev, D_loss, G_loss, ssim_scores, psnr_scores in \
                zip([X_eval_1, X_eval_2, X_eval_3],
                    [Y_eval_1, Y_eval_2, Y_eval_3],
                    [D_loss_test1, D_loss_test2, D_loss_test3],
                    [G_loss_test1, G_loss_test2, G_loss_test3],
                    [ssim_scores1, ssim_scores2, ssim_scores3],
                    [psnr_scores1, psnr_scores2, psnr_scores3]):

            # add losses for fixed test and training batch
            y = np.ones(batch_size)
            g_loss_test = gan.evaluate(x=X_ev[:batch_size], y=[Y_ev[:batch_size], y], verbose=0)

            y = np.ones(2 * batch_size)
            y[:batch_size] = 0
            generated_data = generator.predict_on_batch(X_ev[:batch_size])
            X = np.vstack((generated_data, Y_ev[:batch_size]))
            d_loss_test = discriminator.evaluate(x=X, y=y, verbose=0)

            y = np.ones(batch_size)

            D_loss.append(d_loss_test)
            G_loss.append(g_loss_test)

            generated = generator.predict_on_batch(X_ev)
            ssim_score = metrics.calculate_ssim(Y_ev, generated)
            ssim_scores.append(ssim_score)
            psnr_score = metrics.calculate_psnr(Y_ev, generated)
            psnr_scores.append(psnr_score)

    # Save model after every epoch
    if (epoch % 10 ==0):
        generator.save('generator-'+ str(epoch) + '.h5')
        discriminator.save('discriminator-' + str(epoch) + '.h5')

    # plot images
    f, axarr = plt.subplots(3, 3)
    plt.gcf().set_size_inches(10, 5*3)
    plt.suptitle(str(images_limit) + ' training images per task, epoch ' + str(epoch))

    axarr[0, 0].set_title('out-focus')
    axarr[0, 1].set_title('real in-focus')
    axarr[0, 2].set_title('generated in-focus')

    axarr[0, 0].set_ylabel('task 1')
    axarr[1, 0].set_ylabel('task 2')
    axarr[2, 0].set_ylabel('task 3')

    for i, X_eval, Y_eval in zip(range(3), [X_eval_1, X_eval_2, X_eval_3],
                                 [Y_eval_1, Y_eval_2, Y_eval_3]):

        axarr[i, 0].imshow(X_eval[2,:,:,0])
        axarr[i, 1].imshow(Y_eval[2,:,:,0])
        axarr[i, 2].imshow(generator.predict(X_eval)[2,:,:,0])

    plt.savefig('output-image-' + str(epoch))
    plt.show()

    # dumb the pickle files after every epoch
    with open('data.pkl', 'wb') as f:
        pickle.dump([epoch,
                     D_loss_test1, G_loss_test1,
                     D_loss_test2, G_loss_test2,
                     D_loss_test3, G_loss_test3,
                     ssim_scores1, psnr_scores1,
                     ssim_scores2, psnr_scores2,
                     ssim_scores3, psnr_scores3], f)

pbar.close()
