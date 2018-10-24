
"""
Copyright (c) 2018. All rights reserved.
Created by Resnick Xing on 2018/5/11
"""

import os,cv2
import matplotlib.pyplot as plt
import shutil
import h5py
import numpy as np

def mkdir_if_not_exist(dir_name, is_delete=False):
    """
    创建文件夹
    create dir
    :param dir_name: 文件夹列表
    :param is_delete: 是否删除
    :return: 是否成功
    """
    try:
        if is_delete:
            if os.path.exists(dir_name):
                shutil.rmtree(dir_name)
                print(u'[INFO] Dir "%s" exists, deleting.' % dir_name)

        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(u'[INFO] Dir "%s" not exists, creating.' % dir_name)
        return True
    except Exception as e:
        print('[Exception] %s' % e)
        return False

def load_hdf5(infile):
  with h5py.File(infile,"r") as f:  #"with" close the file after its nested commands
    return f["image"][()]

def write_hdf5(arr,outfile):
  with h5py.File(outfile,"w") as f:
    f.create_dataset("image", data=arr, dtype=arr.dtype)

def genMasks(masks,channels):
    """
    为groundtruth生成相反的掩膜
    generate masks for groundtruth
    :param masks:  groundtruth图
    :param channels: 通道数
    :return:
    """
    assert (len(masks.shape) == 4)
    im_h = masks.shape[2]
    im_w = masks.shape[3]
    masks = np.reshape(masks, (masks.shape[0], channels, im_h * im_w))
    new_masks = np.empty((masks.shape[0], im_h * im_w, channels + 1))

    new_masks[:, :, 0:channels] = masks[:, 0:channels, :].transpose(0, 2, 1)
    maskTotal=np.ma.array(new_masks[:, :, 0], mask=new_masks[:, :, 0]).mask
    for index in range(channels):
        mask = new_masks[:, :, index]
        m = np.ma.array(new_masks[:, :, index], mask=mask)
        maskTotal=maskTotal|m.mask

    new_masks[:, :, channels] = 1 - maskTotal
    return new_masks

def gray2binary(image,threshold=0.5):
    """
    灰度图二值化
    :param image: 灰度图
    :param threshold: 二值化阈值
    :return: 二值图
    """
    image = (image >= threshold) * 1
    return image

def colorize(img,gt,prob):
    image=np.copy(img)
    if np.max(gt)>1:
        gt=gt/255.
    gtlist=[np.where(gt>=0.5)]
    problist = [np.where(prob==1)]
    gtx=gtlist[0][0]
    gty=gtlist[0][1]
    for index in range(gtx.shape[0]):           # gt区域标为绿色
        image[gtx[index],gty[index],0]=0
        image[gtx[index], gty[index], 1] = 1
        image[gtx[index], gty[index], 2] = 0

    probx = problist[0][0]
    proby = problist[0][1]
    for index in range(probx.shape[0]):
        if image[probx[index], proby[index], 1]!=1:    # 预测错误区域标为红色
            image[probx[index], proby[index], 0] = 1
            image[probx[index], proby[index], 1] = 0
            image[probx[index], proby[index], 2] = 0
        else:# 预测正确区域标为蓝色
            image[probx[index], proby[index], 0] = 0
            image[probx[index], proby[index], 1] = 0
            image[probx[index], proby[index], 2] = 1
    return image

def visualize(image,subplot):
    """
    将多张大小相同的图片拼接
    :param image: 图片列表
    :param subplot: 行列数[row,col]
    :return: 拼接图
    """
    row=int(subplot[0])
    col=int(subplot[1])
    height,width=image[0].shape[:2]
    result=np.zeros((height*row,width*col,3))

    total_image=len(image)
    index=0
    for i in range(row):
        for j in range(col):
            row_index=i*height
            col_index=j*width
            if index<total_image:
                try:  #单通道灰度图与3通道彩色图单独处理
                    result[row_index:row_index+height,col_index:col_index+width,:]=image[index]*255
                except:
                    result[row_index:row_index + height, col_index:col_index + width, 0] = image[index]*255
                    result[row_index:row_index + height, col_index:col_index + width, 1] = image[index]*255
                    result[row_index:row_index + height, col_index:col_index + width, 2] = image[index]*255
            index=index+1
    result=result.astype(np.uint8)
    return result



