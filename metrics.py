import numpy as np
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr

def calculate_ssim(X_test, Y_test):
    score = 0.0
    counter = 0

    for i in range(X_test.shape[0]):
        current_score = ssim(255*normalize(np.float64(X_test[i,:,:,0])), 255*normalize(np.float64(Y_test[i,:,:,0])), data_range = 255)
        if (np.isnan(current_score)==False):
            score += current_score
            counter += 1
    score = score / counter
    return score


def calculate_psnr(X_test, Y_test):
    score = 0.0
    counter = 0

    for i in range(X_test.shape[0]):
        current_score = psnr(255*normalize(np.float64(X_test[i,:,:,0])), 255*normalize(np.float64(Y_test[i,:,:,0])), data_range = 255)
        if (np.isnan(current_score)==False):
            score += current_score
            counter += 1

    score = score / counter
    return score



def normalize(x):
    norm_x = (x-np.min(x))/(np.max(x)-np.min(x))
    return norm_x

    

def FT_blur_score(image):

    ft = np.fft.fft(np.fft.fft(image, axis=0), axis=1)
#    ft = np.fft.fftshift(ft)
    ft_abs = np.abs(ft)
    maximum = np.max(ft_abs)
    ft_abs[np.where(ft > (maximum/1000))] = 1
    ft_abs[np.where(ft <= (maximum/1000))] = 0
    th = np.sum(ft_abs)
    #print(th)

    return th / (image.shape[0]*image.shape[1])
