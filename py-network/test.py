import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"  # use GPU-0

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

import datetime
from util.load_cfg import train_cfg, test_cfg,dataset_cfg, sample_cfg
from model.unet import Unet
from data.test.image_patch import padding_images,img2patch_list,patchlist2image,load_test_data
from data.preprocess import preprocess
import matplotlib.pyplot as plt
import cv2
import numpy as np
from glob import glob

testmodel=Unet(sample_cfg["patch_size"])
ckpts = tf.train.Checkpoint(model=testmodel)
ckpts.restore(tf.train.latest_checkpoint(train_cfg["checkpoint_dir"])).expect_partial()


def test_function(image_path,test_mask_dir,test_save_dir):
    image_name = image_path.split("/")[-1].split("_")[0]

    # load and process test images （加载并预处理图像）
    image = plt.imread(image_path)
    original_shape = image.shape
    mask = plt.imread(test_mask_dir + image_name + "_test_mask.gif")
    mask = np.where(mask > 0, 1, 0)

    # image to patches （图像分块）
    image, pad_mask = padding_images(image, mask,test_cfg["stride"])
    image = preprocess(image, pad_mask)
    test_patch_list = img2patch_list(image,test_cfg["stride"])

    # test dataloader （整合测试数据）
    test_dataset = tf.data.Dataset.from_tensor_slices(test_patch_list)
    test_dataset = test_dataset.map(load_test_data)
    test_dataset = test_dataset.batch(64)
    pred_result = []

    # test process （测试）
    print("testing image:", int(image_name))
    for batch, patch in enumerate(test_dataset):
        _, pred = testmodel(patch, training=False)

        pred = pred.numpy()
        pred_result.append(pred)
    pred_result = np.concatenate(pred_result, axis=0)

    # patches to image （还原图像）
    #print("post processing:", image_name)
    pred_image = patchlist2image(pred_result, test_cfg["stride"],image.shape)

    pred_image = pred_image[:original_shape[0], :original_shape[1]]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))  # 定义结构元素的形状和大小
    mask = cv2.erode(mask.astype(np.uint8), kernel)  # 膨胀操作

    pred_image = pred_image * mask
    pred_image = np.where(pred_image > test_cfg["threshold"], 1, 0)

    # visualize the test result （可视化测试结果）
    plt.figure(figsize=(8, 8))
    plt.title(image_name + "-(" + str(image.shape[0]) + "," + str(image.shape[1]) + ")")
    plt.imshow(pred_image, cmap=plt.cm.gray)
    plt.show()

    plt.imsave(test_save_dir + str(int(image_name)) + ".png", pred_image, cmap=plt.cm.gray)


if __name__=="__main__":
    test_dir=dataset_cfg["dataset_path"]+dataset_cfg["test_dir"]
    test_image_dir = test_dir + dataset_cfg["test_image_dir"]
    test_mask_dir = test_dir + dataset_cfg["test_mask_dir"]
    test_groundtruth_dir = test_dir + dataset_cfg["test_groundtruth_dir"]
    test_save_dir = test_dir + dataset_cfg["test_save_dir"]

    test_image_path_list = sorted(glob(test_image_dir + "*.tif"))

    for i in range(len(test_image_path_list)):
        image_path = test_image_path_list[i]
        test_function(image_path,test_mask_dir,test_save_dir)