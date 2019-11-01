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

def get_n_fake(lim, alpha):
    return int(lim/alpha - lim)

n_fake = get_n_fake(images_limit, alpha)
print("We're getting ", n_fake, " extra fake images")


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

if os.path.isdir("CIN-GAN_results"):
    shutil.rmtree("CIN-GAN_results")
os.mkdir("CIN-GAN_results")
os.chdir("./CIN-GAN_results")



# ========================= Create models and compile ======================= #
# Following parameter and optimizer set as recommended in paper
n_critic = 5
optimizer = RMSprop(lr=0.00005)
critic_opt = Adam(lr=1e-5, beta_1=0.5, beta_2=0.9)
gen_opt = Adam(lr=1e-5, beta_1=0.5, beta_2=0.9)

# Build the generator and critic
generator = network_blocks.build_CIN_generator()
critic = network_blocks.build_CIN_critic()
critic_model = network_blocks.CIN_Critic_model(generator, critic)
generator_model = network_blocks.CIN_Generator_model(generator, critic)



def save_images(images_array, name):
    plt.figure(figsize=(9, 9))
    for i in range(9):
        plt.subplot(3, 3, i+1)
        img = images_array[i, :]
        img = img.reshape((new_height, new_width))
        plt.imshow(img)
        plt.axis('off')
    plt.savefig(str(name) + '.png')
    plt.close()

epochs = 120
sample_interval = 100

# fix a batch for testing purpose
X_eval_1 = X1[1 * batch_size: (1 + 1) * batch_size]
Y_eval_1 = Y1[1 * batch_size: (1 + 1) * batch_size]

X_eval_2 = X2[1 * batch_size: (1 + 1) * batch_size]
Y_eval_2 = Y2[1 * batch_size: (1 + 1) * batch_size]

X_eval_3 = X3[1 * batch_size: (1 + 1) * batch_size]
Y_eval_3 = Y3[1 * batch_size: (1 + 1) * batch_size]

# Adversarial ground truths
valid = -np.ones((batch_size, 1))
fake =  np.ones((batch_size, 1))
dummy = np.zeros((batch_size, 1)) # Dummy gt for gradient penalty

# store losses
D_loss_test = []
G_loss_test = []
D_loss_train = []
G_loss_train = []
ssim_scores = []
psnr_scores = []
fourier_scores = []

num_batches = int(X1.shape[0] / batch_size)
pbar = tqdm(total=epochs * num_batches)

# save the initial image
save_images(Y_eval_1[:9], 'start-1')
save_images(X_eval_1[:9], 'end-1')


for epoch in range(1, epochs+1):

    np.random.shuffle(X1)
    np.random.shuffle(Y1)
    np.random.shuffle(X2)
    np.random.shuffle(Y2)
    np.random.shuffle(X3)
    np.random.shuffle(Y3)


    for i in range(num_batches):

        pbar.update(1)

        task = np.random.choice(3) + 1

        if task == 1:
            K.set_value(c1, [1])
            K.set_value(c2, [0])
            K.set_value(c3, [0])
            X_train = X1
            Y_train = Y1


        elif task == 2:
            K.set_value(c1, [0])
            K.set_value(c2, [1])
            K.set_value(c3, [0])
            X_train = X2
            Y_train = Y2

        else:
            K.set_value(c1, [0])
            K.set_value(c2, [0])
            K.set_value(c3, [1])
            X_train = X3
            Y_train = Y3

        for _ in range(n_critic):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(i, (i + 1) * batch_size, batch_size)
            focused_imgs = Y_train[idx]

            # Sample generator input
            unfocused_imgs = X_train[idx]

            # Train the critic
            d_loss = critic_model.train_on_batch([unfocused_imgs, focused_imgs],
                                                 [valid, fake, dummy])

        # ---------------------
        #  Train Generator
        # ---------------------

        # focused images and unfocused images, train the generator
        unfocused_imgs = X_train[i * batch_size: (i + 1) * batch_size]
        focused_imgs = Y_train[i * batch_size: (i + 1) * batch_size]
        g_loss = generator_model.train_on_batch(focused_imgs, [unfocused_imgs, unfocused_imgs, valid])


#        # add losses for fixed test and training batch
        d_loss_test = critic_model.evaluate([Y_eval_1, X_eval_1], [fake, valid, dummy], verbose=0)
#        d_loss_train = critic_model.evaluate([Y_eval_train, X_eval_train], [valid, fake, dummy], verbose=0)
        g_loss_test = generator_model.evaluate(Y_eval_1, [X_eval_1, X_eval_1, valid], verbose=0)
#        g_loss_train = generator_model.evaluate(X_eval_train, [Y_eval_train, Y_eval_train, valid], verbose=0)
#
        D_loss_test.append(d_loss_test)
        G_loss_test.append(g_loss_test)
#        D_loss_train.append(d_loss_train)
#        G_loss_train.append(g_loss_train)

#        ## calculate scores
#        generated = generator.predict_on_batch(X_eval_test)
#        ssim_score = calculate_ssim(Y_eval_test, generated)
#        ssim_scores.append(ssim_score)
#        psnr_score = calculate_psnr(Y_eval_test, generated)
#        psnr_scores.append(psnr_score)
#        fourier_score = calculate_fourier(generated)
#        fourier_scores.append(fourier_score)

        with open('loss.pkl', 'wb') as f:
            pickle.dump([D_loss_train, G_loss_train, D_loss_test, G_loss_test,
                         ssim_scores, psnr_scores, fourier_scores], f)

        # If at save interval => save generated image samples
        if i % sample_interval == 0:

            # Plot the progress
            print ("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss[0]))
#            with open('loss-iter.pkl', 'wb') as f:
#                pickle.dump([D_loss_train, G_loss_train, D_loss_test, G_loss_test,
#                         ssim_scores, psnr_scores, fourier_scores], f)

            # Plot losses
            fig = plt.figure(figsize=(10, 5))
            fig.suptitle('epoch: ' + str(epoch + 1))

            plt.plot(np.array(D_loss_test)[:,1], label="discriminator's Wass fake loss", color='b')
            plt.plot(np.array(D_loss_test)[:,2], label="discriminator's Wass valid loss", color='darkblue')
            plt.plot(np.array(G_loss_test)[:,1], label="generator's content VGG16 loss", color='r')
            plt.plot(np.array(G_loss_test)[:,2], label="generator's style VGG19 loss", color='orange')
            #plt.plot(np.array(gan_loss)[:,1], label="generator's Wasserstein loss", color='magenta')
            plt.xlim([0, epoch * num_batches + i])
            plt.legend()
            plt.savefig('losses.png')

            plt.ylim([min(np.array(D_loss_test)[:,0]), max(np.array(D_loss_test)[:,0])])
            plt.savefig('losses_zoomed_in.png')
            plt.close()


            # save images every iter
            K.set_value(c1, [1])
            K.set_value(c2, [0])
            K.set_value(c3, [0])
            save_images(generator.predict_on_batch(Y_eval_1[:9]), 'data1_'+str(epoch))

            K.set_value(c1, [0])
            K.set_value(c2, [1])
            K.set_value(c3, [0])
            save_images(generator.predict_on_batch(Y_eval_2[:9]), 'data2_'+str(epoch))

            K.set_value(c1, [0])
            K.set_value(c2, [0])
            K.set_value(c3, [1])
            save_images(generator.predict_on_batch(Y_eval_3[:9]), 'data3_'+str(epoch))

    # store the model
    generator.save('generator-' + str(epoch) + '.h5')
    critic.save('critic-' + str(epoch) + '.h5')


pbar.close()
