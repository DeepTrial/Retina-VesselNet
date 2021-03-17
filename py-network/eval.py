import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"  # use GPU-0

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

import datetime
from util.load_cfg import train_cfg, test_cfg,dataset_cfg, sample_cfg
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from util.dice import *

def eval_function(test_groundtruth_path_list):
    dice_list = []
    roc_list = []
    pr_list = []
    tpr_list = []
    tnr_list = []
    sum_tp = 0
    sum_tn = 0
    sum_fp = 0
    sum_fn = 0

    roc = tf.keras.metrics.AUC(num_thresholds=200, curve="ROC")
    pr = tf.keras.metrics.AUC(num_thresholds=200, curve="PR")
    recall = tf.keras.metrics.Recall()

    tn = tf.keras.metrics.TrueNegatives()
    tp = tf.keras.metrics.TruePositives()
    fn = tf.keras.metrics.FalseNegatives()
    fp = tf.keras.metrics.FalsePositives()

    for idx in range(len(test_groundtruth_path_list)):
        name = test_groundtruth_path_list[idx].split("/")[-1].split(".")[0].split("_")[0]
        roc.reset_states()
        pr.reset_states()
        tn.reset_states()
        fn.reset_states()
        tp.reset_states()
        fp.reset_states()

        groundtruth = plt.imread(test_groundtruth_path_list[idx])
        preds = plt.imread(test_save_dir + str(int(name)) + ".png")
        groundtruth = np.array(groundtruth, dtype=np.float32)
        groundtruth = groundtruth / 255.0

        roc.update_state(groundtruth, preds[:, :, 0])  # png image is 4-channel
        pr.update_state(groundtruth, preds[:, :, 0])
        tn.update_state(groundtruth, preds[:, :, 0])
        tp.update_state(groundtruth, preds[:, :, 0])
        fn.update_state(groundtruth, preds[:, :, 0])
        fp.update_state(groundtruth, preds[:, :, 0])

        dice_list.append(dice(groundtruth, preds[:, :, 0]).numpy())
        roc_list.append(roc.result().numpy())
        pr_list.append(pr.result().numpy())

        current_tn = tn.result().numpy()
        current_tp = tp.result().numpy()
        current_fn = fn.result().numpy()
        current_fp = fp.result().numpy()

        sum_tp += current_tp
        sum_tn += current_tn
        sum_fp += current_fp
        sum_fn += current_fn

        tpr_list.append(current_tp / (current_tp + current_fn))
        tnr_list.append(current_tn / (current_tn + current_fp))

    print("average dice score for all predict vessel masks:", np.mean(dice_list))
    print("average AUC for all predict vessel masks:", np.mean(roc_list))
    print("average PR for all predict vessel masks:", np.mean(pr_list))
    print("average recall(sensitivity) for all predict vessel masks:", np.mean(tpr_list))
    print("average specificity for all predict vessel masks:", np.mean(tnr_list))

    return sum_tp, sum_fn,sum_fp, sum_tn


def confusion_matrix(sum_tp, sum_fn,sum_fp, sum_tn):
    actual = ["Positive", "Negative"]
    classes = list(set(actual))
    classes.sort(reverse=True)
    confusion_matrix = [[sum_tp, sum_fn], [sum_fp, sum_tn]]
    print("confusion matrix", confusion_matrix)

    plt.figure(figsize=(8, 8))
    font_size = 18
    plt.imshow(confusion_matrix, cmap=plt.cm.Blues)
    indices = range(len(confusion_matrix))
    plt.xticks(indices, classes, rotation=40, fontsize=font_size)
    plt.yticks([0.00, 1.00], classes, fontsize=font_size)
    plt.ylim(1.5, -0.5)

    plt.title("Confusion matrix (Pixel Level)", fontdict={'weight': 'normal', 'size': font_size})
    plt.xlabel('Predict Label', fontsize=font_size)
    plt.ylabel('Actual Label', fontsize=font_size)

    for first_index in range(len(confusion_matrix)):
        for second_index in range(len(confusion_matrix[first_index])):
            if confusion_matrix[first_index][second_index] > 2e6:
                text_color = "w"
            else:
                text_color = "black"
            plt.text(first_index, second_index, confusion_matrix[first_index][second_index], fontsize=font_size,color=text_color, verticalalignment='center', horizontalalignment='center', )

    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=font_size)
    plt.show()

if __name__=="__main__":
    test_dir = dataset_cfg["dataset_path"] + dataset_cfg["test_dir"]
    test_image_dir = test_dir + dataset_cfg["test_image_dir"]
    test_mask_dir = test_dir + dataset_cfg["test_mask_dir"]
    test_groundtruth_dir = test_dir + dataset_cfg["test_groundtruth_dir"]
    test_save_dir = test_dir + dataset_cfg["test_save_dir"]

    test_groundtruth_path_list = sorted(glob(test_groundtruth_dir + "*.gif"))
    test_pred_path_list = sorted(glob(test_save_dir + "*.png"))


    assert len(test_groundtruth_path_list)==len(test_pred_path_list)

    sum_tp, sum_fn,sum_fp, sum_tn=eval_function(test_groundtruth_path_list)
    confusion_matrix(sum_tp, sum_fn,sum_fp, sum_tn)