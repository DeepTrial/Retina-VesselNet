import numpy as np
import tensorflow as tf
from util.load_cfg import sample_cfg

patch_size=sample_cfg["patch_size"]

def padding_images(image, mask,stride):
    h, w = image.shape[:2]
    new_h, new_w = h, w
    while (new_h - patch_size) % stride != 0:
        new_h += 1
    while (new_w - patch_size) % stride != 0:
        new_w += 1
    pad_image = np.zeros((new_h, new_w, 3))
    pad_image[:h, :w, :] = image

    pad_mask = np.zeros((new_h, new_w))
    pad_mask[:h, :w] = mask

    return pad_image, pad_mask


# images to patches （将图像分块）
def img2patch_list(image ,stride):
    patch_list = []
    # image_binary=0.8*image[:,:,1:2]+0.2*image[:,:,2:3]
    for j in range(0, image.shape[1] - patch_size + 1, stride):
        for i in range(0, image.shape[0] - patch_size + 1, stride):
            patch = image[i:i + patch_size, j:j + patch_size, :]
            patch_list.append(patch)
    return patch_list


# patches to image （将图像块拼成图像）
def patchlist2image(patch_list, stride, image_shape):
    result = np.zeros(image_shape[:2])
    sum_matrix = np.zeros(image_shape[:2])
    index_x, index_y = 0, 0
    for i in range(patch_list.shape[0]):
        patch = patch_list[i, :, :, 0]
        # patch=np.where(patch>0.5,1,0)
        # print(patch)
        result[index_x:index_x + patch_size, index_y:index_y + patch_size] += patch
        sum_matrix[index_x:index_x + patch_size, index_y:index_y + patch_size] += 1
        index_x += stride
        if index_x + patch_size > image_shape[0]:
            index_x = 0
            index_y += stride
    return result / sum_matrix


def load_test_data(image):
  #image=tf.image.decode_jpeg(image,channels=1)
  #print(image.shape)
  image=tf.image.resize(image,[patch_size,patch_size])
  #image/=255.0
  return image