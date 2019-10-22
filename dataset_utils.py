import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

def normalize_split(X, Y, images_limit, eval_limit=30):

    # normalize the data
    np.random.seed(42)
    np.random.shuffle(X)
    np.random.seed(42)
    np.random.shuffle(Y)

    X_train = X[:images_limit]
    Y_train = Y[:images_limit]
    X_test = X[-eval_limit:]
    Y_test = Y[-eval_limit:]

    minYTrain = np.min(Y_train)
    maxYTrain = np.max(Y_train)
    minXTrain = np.min(X_train)
    maxXTrain = np.max(X_train)

    Y_train = 2*((Y_train - minYTrain) / (maxYTrain - minYTrain)) -1
    X_train = 2*((X_train - minXTrain) / (maxXTrain - minXTrain))-1

    Y_test = 2*((Y_test - minYTrain) / (maxYTrain - minYTrain)) -1
    X_test = 2*((X_test - minXTrain) / (maxXTrain - minXTrain))-1

    return X_train, Y_train, X_test, Y_test

# shuffle in unison
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    np.random.seed(42)
    p = np.random.permutation(len(a))
    return a[p], b[p]




def print_9_images(images_array):
    plt.figure(figsize=(9, 9))
    for i in range(9):
        plt.subplot(3, 3, i+1)
        img = images_array[i, :]
        img = img.reshape((new_height, new_width))
        plt.imshow(img)
        plt.axis('off')
    plt.savefig(str(epoch + 1) + '.png')
    plt.close()
