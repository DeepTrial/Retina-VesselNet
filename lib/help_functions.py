import h5py
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import cv2
import configparser
config = configparser.RawConfigParser()
config.read('./configuration.txt')
patch_height = int(config.get('data attributes', 'patch_height'))
patch_width = int(config.get('data attributes', 'patch_width'))
num_lesion_class = int(config.get('data attributes', 'num_lesion_class'))

def load_hdf5(infile):
  with h5py.File(infile,"r") as f:  #"with" close the file after its nested commands
    return f["image"][()]

def write_hdf5(arr,outfile):
  with h5py.File(outfile,"w") as f:
    f.create_dataset("image", data=arr, dtype=arr.dtype)

#convert RGB image in black and white
def rgb2gray(rgb):
    assert (len(rgb.shape)==4)  #4D arrays
    assert (rgb.shape[1]==3)
    bn_imgs = rgb[:,0,:,:]*0.299 + rgb[:,1,:,:]*0.587 + rgb[:,2,:,:]*0.114
    bn_imgs = np.reshape(bn_imgs,(rgb.shape[0],1,rgb.shape[2],rgb.shape[3]))
    return bn_imgs

#group a set of images row per columns
def group_images(data,per_row):
    assert data.shape[0]%per_row==0
    #assert (data.shape[1]==1 or data.shape[1]==3)
    data = np.transpose(data,(0,2,3,1))  #corect format for imshow
    all_stripe = []
    for i in range(int(data.shape[0]/per_row)):
        stripe = data[i*per_row]
        for k in range(i*per_row+1, i*per_row+per_row):
            stripe = np.concatenate((stripe,data[k]),axis=1)
        all_stripe.append(stripe)
    totimg = all_stripe[0]
    for i in range(1,len(all_stripe)):
        totimg = np.concatenate((totimg,all_stripe[i]),axis=0)
    return totimg

def visualize_all(data,filename):
    assert(len(data.shape)==3)
    if data.shape[2]==1:
        data=np.reshape(data,(data.shape[0],data.shape[1]))
        return visualize(data,filename,-1)
    elif data.shape[2]==3:
        return visualize(data,filename,-1)
    elif data.shape[2]>3:
        for i in range(data.shape[2]):
            return visualize(data[:,:,i],filename,i)

#visualize image (as PIL image, NOT as matplotlib!)
def visualize(data,filename,num):
    img = None
    if np.max(data)>1:
        img = Image.fromarray(data.astype(np.uint8))   #the image is already 0-255
    else:
        img = Image.fromarray((data*255).astype(np.uint8))  #the image is between 0-1
    if num>=0:
        filename=filename+str(num)
    img.save(filename)
    return img


#prepare the mask in the right shape for the Unet
#num_lesion_class=1
def masks_Unet(masks,num_lesion_class):
    assert (len(masks.shape)==4)  #4D arrays
    #assert (masks.shape[1]==1 )  #check the channel is 1
    im_h = masks.shape[2]
    im_w = masks.shape[3]
    masks = np.reshape(masks,(masks.shape[0],num_lesion_class,im_h*im_w))
    new_masks = np.empty((masks.shape[0],im_h*im_w,num_lesion_class+1))

    new_masks[:,:,0:num_lesion_class]=masks[:,0:num_lesion_class,:].transpose(0,2,1)
    #for i in range(4):
    mask0=new_masks[:,:,0]
    m0=np.ma.array(new_masks[:,:,0],mask=mask0)

    # mask1=new_masks[:,:,1]
    # m1=np.ma.array(new_masks[:,:,1],mask=mask1)
    #
    # mask2=new_masks[:,:,2]
    # m2=np.ma.array(new_masks[:,:,2],mask=mask2)
    #
    # mask3=new_masks[:,:,3]
    # m3=np.ma.array(new_masks[:,:,3],mask=mask3)


    new_masks[:,:,num_lesion_class]=1-(m0.mask)#|m1.mask|m2.mask|m3.mask)
    #for i in range(masks.shape[0]):
    #    for j in range(im_h*im_w):
    #        cnt=0
    #        for k in range(4):
    #            if  masks[i,k,j] == 0:
    #                new_masks[i,j,k]=0
    #                cnt=cnt+1
    #            else:
    #                new_masks[i,j,k]=1
    #        if cnt==4:
    #            new_masks[i,j,4]=1
    return new_masks


def pred_to_imgs(pred,mode="original"):#1
    assert (len(pred.shape)==3)  #3D array: (Npatches,height*width,2)
    #assert (pred.shape[2]==2 )  #check the classes are 2
    pred_images = np.empty((pred.shape[0],pred.shape[1],3))  #(Npatches,height*width)
    if mode=="original":
        pred_images[:,:,0:3]=pred[:,:,0:3]
        pred_images[:,:,2]+=pred[:,:,3]
        pred_images[:,:,2]/=np.max(pred_images[:,:,2])
        #pred_images[:,:,1]=pred[:,:,3]
        #for i in range(pred.shape[0]):
        #    for pix in range(pred.shape[1]):
        #        if pred[i,pix,3]>0.5:
        #            pred_images[i,pix,0:3]=1.0
                #pred_images[i,pix]=pred_images[i,pix]+pred[i,pix,3]
    elif mode=="threshold":
        for i in range(pred.shape[0]):
            for pix in range(pred.shape[1]):
                if pred[i,pix,1]>=0.1:
                    pred_images[i,pix]=1
                else:
                    pred_images[i,pix]=0
    else:
        print("mode " +str(mode) +" not recognized, it can be 'original' or 'threshold'")
        exit()
    pred_images = np.reshape(pred_images,(pred_images.shape[0],patch_height,patch_width,3))
    return pred_images

def pred_to_imgs_3(pred,mode="original"):
    assert (len(pred.shape)==3)  #3D array: (Npatches,height*width,2)
    #assert (pred.shape[2]==2 )  #check the classes are 2
    pred_images = np.empty((pred.shape[0],pred.shape[1],3))  #(Npatches,height*width)
    if mode=="original":
        pred_images[:,:,0:3]=pred[:,:,0:3]
        #pred_images[:,:,1]=pred[:,:,3]
        #for i in range(pred.shape[0]):
        #    for pix in range(pred.shape[1]):
        #        if pred[i,pix,3]>0.5:
        #            pred_images[i,pix,0:3]=1.0
                #pred_images[i,pix]=pred_images[i,pix]+pred[i,pix,3]
    elif mode=="threshold":
        for i in range(pred.shape[0]):
            for pix in range(pred.shape[1]):
                if pred[i,pix,1]>=0.5:
                    pred_images[i,pix]=1
                else:
                    pred_images[i,pix]=0
    else:
        print("mode " +str(mode) +" not recognized, it can be 'original' or 'threshold'")
        exit()
    pred_images = np.reshape(pred_images,(pred_images.shape[0],patch_height,patch_width,3))
    return pred_images

def pred_to_imgs(pred,mode="original"):
    assert (len(pred.shape)==3)  #3D array: (Npatches,height*width,2)
    #assert (pred.shape[2]==2 )  #check the classes are 2
    pred_images=None   #(Npatches,height*width)
    if mode=="original":
        pred_images=pred[:,:,0:num_lesion_class+1]
        #pred_images[:,:,1]=pred[:,:,3]
        #for i in range(pred.shape[0]):
        #    for pix in range(pred.shape[1]):
        #        if pred[i,pix,3]>0.5:
        #            pred_images[i,pix,0:3]=1.0
                #pred_images[i,pix]=pred_images[i,pix]+pred[i,pix,3]
    elif mode=="threshold":
        for i in range(pred.shape[0]):
            for pix in range(pred.shape[1]):
                if pred[i,pix,1]>=0.5:
                    pred_images[i,pix]=1
                else:
                    pred_images[i,pix]=0
    else:
        print("mode " +str(mode) +" not recognized, it can be 'original' or 'threshold'")
        exit()
    pred_images = np.reshape(pred_images,(pred_images.shape[0],patch_height,patch_width,num_lesion_class+1))
    return pred_images

def pred_to_imgs_4(pred,mode="original"):
    assert (len(pred.shape)==3)  #3D array: (Npatches,height*width,2)
    #assert (pred.shape[2]==2 )  #check the classes are 2
    pred_images = np.empty((pred.shape[0],pred.shape[1],4))  #(Npatches,height*width)
    if mode=="original":
        pred_images[:,:,0:4]=pred[:,:,0:4]
        #pred_images[:,:,2]+=pred[:,:,3]
        #pred_images[:,:,2]/=np.max(pred_images[:,:,2])
        #pred_images[:,:,1]=pred[:,:,3]
        #for i in range(pred.shape[0]):
        #    for pix in range(pred.shape[1]):
        #        if pred[i,pix,3]>0.5:
        #            pred_images[i,pix,0:3]=1.0
                #pred_images[i,pix]=pred_images[i,pix]+pred[i,pix,3]
    elif mode=="threshold":
        for i in range(pred.shape[0]):
            for pix in range(pred.shape[1]):
                if pred[i,pix,1]>=0.5:
                    pred_images[i,pix]=1
                else:
                    pred_images[i,pix]=0
    else:
        print("mode " +str(mode) +" not recognized, it can be 'original' or 'threshold'")
        exit()
    pred_images = np.reshape(pred_images,(pred_images.shape[0],patch_height,patch_width,4))
    return pred_images
