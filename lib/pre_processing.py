###################################################
#
#   Script to pre-process the original imgs
#
##################################################


import numpy as np
from PIL import Image
import cv2
from lib.help_functions import *


#My pre processing (use for both training and testing!)
def my_PreProc(data):
    assert(len(data.shape)==4)
    assert (data.shape[1]==1)  #Use the original images
    #black-white conversion
    #train_imgs = rgb2gray(data)
    train_imgs=np.zeros(data.shape)
    train_imgs0=np.zeros([data.shape[0],1,data.shape[2],data.shape[3]])
    #train_imgs1=np.zeros([data.shape[0],1,data.shape[2],data.shape[3]])
    #train_imgs2=np.zeros([data.shape[0],1,data.shape[2],data.shape[3]])
    train_imgs0[:,0,:,:]=data[:,0,:,:]
    #train_imgs1[:,0,:,:]=data[:,1,:,:]
    #train_imgs2[:,0,:,:]=data[:,2,:,:]

    #train_imgs = data
    #my preprocessing:
#    train_imgs = dataset_normalized(train_imgs)
#    train_imgs = clahe_equalized(train_imgs)
#    train_imgs = adjust_gamma(train_imgs, 1.2)
#    train_imgs = train_imgs/255.  #reduce to 0-1 range
    train_imgs0 = dataset_normalized(train_imgs0)
    train_imgs0 = clahe_equalized(train_imgs0)
    train_imgs0 = adjust_gamma(train_imgs0, 1.2)
    train_imgs0 = train_imgs0/255.  #reduce to 0-1 range
    # train_imgs1 = dataset_normalized(train_imgs1)
    # train_imgs1 = clahe_equalized(train_imgs1)
    # train_imgs1 = adjust_gamma(train_imgs1, 1.2)
    # train_imgs1 = train_imgs1/255.  #reduce to 0-1 range
    # train_imgs2 = dataset_normalized(train_imgs2)
    # train_imgs2 = clahe_equalized(train_imgs2)
    # train_imgs2 = adjust_gamma(train_imgs2, 1.2)
    # train_imgs2 = train_imgs2/255.  #reduce to 0-1 range
    train_imgs[:,0,:,:]=train_imgs0[:,0,:,:]
    #train_imgs[:,1,:,:]=train_imgs1[:,0,:,:]
    #train_imgs[:,2,:,:]=train_imgs2[:,0,:,:]
    return train_imgs


#============================================================
#========= PRE PROCESSING FUNCTIONS ========================#
#============================================================

#==== histogram equalization
def histo_equalized(imgs):
    imgs_equalized = np.empty(imgs.shape)
    imgs_equalized = cv2.equalizeHist(np.array(imgs, dtype = np.uint8))
    return imgs_equalized


# CLAHE (Contrast Limited Adaptive Histogram Equalization)
#adaptive histogram equalization is used. In this, image is divided into small blocks called "tiles" (tileSize is 8x8 by default in OpenCV). Then each of these blocks are histogram equalized as usual. So in a small area, histogram would confine to a small region (unless there is noise). If noise is there, it will be amplified. To avoid this, contrast limiting is applied. If any histogram bin is above the specified contrast limit (by default 40 in OpenCV), those pixels are clipped and distributed uniformly to other bins before applying histogram equalization. After equalization, to remove artifacts in tile borders, bilinear interpolation is applied
def clahe_equalized(imgs):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    #create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i,0] = clahe.apply(np.array(imgs[i,0], dtype = np.uint8))
    return imgs_equalized


# ===== normalize over the dataset
def dataset_normalized(imgs):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    imgs_normalized = np.empty(imgs.shape)
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs-imgs_mean)/imgs_std
    for i in range(imgs.shape[0]):
        imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (np.max(imgs_normalized[i])-np.min(imgs_normalized[i])))*255
    return imgs_normalized


def adjust_gamma(imgs, gamma=1.0):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    new_imgs = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        new_imgs[i,0] = cv2.LUT(np.array(imgs[i,0], dtype = np.uint8), table)
    return new_imgs
