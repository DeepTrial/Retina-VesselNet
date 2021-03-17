import tensorflow as tf
import random
import os
import cv2
import matplotlib.pyplot as plt
from glob import glob
from data.preprocess import preprocess
import shutil
from tqdm import tqdm
from sklearn.utils import shuffle
from data.train.image2patch import image2patch
from util.load_cfg import sample_cfg


patch_size=sample_cfg["patch_size"]
patch_threshold=sample_cfg["patch_threshold"]
patch_num=sample_cfg["patch_num"]
patch_threshold=sample_cfg["patch_threshold"]
train_val_rate=sample_cfg["train_val_rate"]

def load_image_groundtruth(img_path, groundtruth_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [patch_size, patch_size])

    groundtruth = tf.io.read_file(groundtruth_path)
    groundtruth = tf.image.decode_jpeg(groundtruth, channels=1)

    # data argument (数据增强部分)
    if random.uniform(0, 1) >= 0.5:
        img = tf.image.flip_left_right(img)
        groundtruth = tf.image.flip_left_right(groundtruth)

    #   if random.uniform(0,1)>=0.5:
    #     seeds=random.uniform(0,1)
    #     img=tf.image.central_crop(img,seeds)
    #     groundtruth=tf.image.central_crop(groundtruth,seeds)

    img = tf.image.resize(img, [patch_size, patch_size])
    groundtruth = tf.image.resize(groundtruth, [patch_size, patch_size])

    img /= 255.0
    groundtruth = (groundtruth + 40) / 255.0
    groundtruth = tf.cast(groundtruth, dtype=tf.uint8)

    return img, groundtruth



def get_dataset(dataset_cfg,regenerate=True):
    ## load configs
    dataset_path = dataset_cfg["dataset_path"]  # modify the dataset_path to your own dir（将dataset_path修改至你自己的路径）
    train_dir = dataset_path + dataset_cfg["train_dir"]
    train_image_dir = train_dir + dataset_cfg["train_image_dir"]
    train_mask_dir = train_dir + dataset_cfg["train_mask_dir"]
    train_groundtruth_dir = train_dir + dataset_cfg["train_groundtruth_dir"]
    train_patch_dir = train_dir + dataset_cfg["train_patch_dir"]

    # scan the training image
    train_image_path_list = glob(train_image_dir + "*.tif")

    #split training/val set
    val_image_path_list = random.sample(train_image_path_list, int(len(train_image_path_list) * (1 - train_val_rate)))
    train_image_path_list = [i for i in train_image_path_list if i not in val_image_path_list]

    print("[INFO] Number of training images:", len(train_image_path_list))
    print("[INFO] Number of valid images:", len(val_image_path_list))

    if not os.path.exists(train_patch_dir):
        os.mkdir(train_patch_dir)
    
    # generate patch images (生成图像块数据)
    if regenerate:
        shutil.rmtree(train_patch_dir)
        os.mkdir(train_patch_dir)
        
        for i in tqdm(range(len(train_image_path_list)), desc="Generate the training patches: "):
            image2patch(train_image_path_list[i], patch_num, patch_size,patch_threshold,train_patch_dir,train_groundtruth_dir,train_mask_dir, training=True,show=False)  # set show=True to visualize the sample process, which is much slower than show=False

        for i in tqdm(range(len(val_image_path_list)), desc="Generate the val patches: "):
            image2patch(val_image_path_list[i], patch_num, patch_size,patch_threshold,train_patch_dir,train_groundtruth_dir,train_mask_dir, training=False,show=False)  # set show=True to visualize the sample process, which is much slower than show=False
    else:
        print("[INFO] use generated patches...")
    # load training/val patches
    train_patch_img_path_list = sorted(glob(train_patch_dir + "*-*-img.jpg"))
    train_patch_groundtruth_path_list = sorted(glob(train_patch_dir + "*-*-groundtruth.jpg"))
    train_patch_img_path_list, train_patch_groundtruth_path_list = shuffle(train_patch_img_path_list,train_patch_groundtruth_path_list,random_state=0)

    # make sure that img-list and mask-list is in order (确保打乱后的image-mask还是对应的)
    # print(len(train_patch_img_path_list), len(train_patch_groundtruth_path_list))
    # print(train_patch_img_path_list[:2])
    # print(train_patch_groundtruth_path_list[:2])

    val_patch_img_path_list = sorted(glob(train_patch_dir + "*_*_val_img.jpg"))
    val_patch_groundtruth_path_list = sorted(glob(train_patch_dir + "*_*_val_groundtruth.jpg"))

    # print(val_patch_img_path_list[:2])
    # print(val_patch_groundtruth_path_list[:2])

    # Training Dataloader
    train_dataset = tf.data.Dataset.from_tensor_slices((train_patch_img_path_list, train_patch_groundtruth_path_list))
    train_dataset = train_dataset.map(load_image_groundtruth, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #train_dataset = train_dataset.shuffle(buffer_size=1300).prefetch(BATCH_SIZE).batch(BATCH_SIZE)

    # VAL Dataloader
    val_dataset = tf.data.Dataset.from_tensor_slices((val_patch_img_path_list, val_patch_groundtruth_path_list))
    val_dataset = val_dataset.map(load_image_groundtruth, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #val_dataset = val_dataset.shuffle(buffer_size=1300).prefetch(BATCH_SIZE).batch(BATCH_SIZE)

    return train_dataset,val_dataset