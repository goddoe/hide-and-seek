import pickle

import numpy as np
import matplotlib.pyplot as plt

from libs.image_utils import (calc_iou_accuracy,
                              calc_iou_top_1_accuracy,
                              draw_bounding_box)
"""
Helper
"""


def evaluate(model, X, P, Y_one_hot, name, thresh_cam=0.5, thresh_iou=0.5):
    cam_list = model.calc_cam(X, Y_one_hot)
    bbox_list = model.location(cam_list=cam_list, thresh=0.5)
    Y_pred = model.predict(X)
    Y_pred = np.argmax(Y_pred, axis=1)
    Y_real = np.argmax(Y_one_hot, axis=1)
    
    gt_known_loc_accuracy = calc_iou_accuracy(
        bbox_list, P, thresh_iou=thresh_iou)
    top_1_loc_accuracy = calc_iou_top_1_accuracy(
        bbox_list, Y_pred, P, Y_real)
    print("GT-known-Loc {} iou_accuracy : {}".format(name, gt_known_loc_accuracy))
    print("Top-1 Loc {} iou_accuracy : {}".format(name, top_1_loc_accuracy))
    return cam_list, bbox_list, gt_known_loc_accuracy, top_1_loc_accuracy

def visualize(X, P, Y, cam_list, bbox_list, idx_word_dict, n_show=3, start=0):
    fig, axs = plt.subplots(3, 2, figsize=(6, 18))
    for i in range(n_show):
        idx = start + i 
        loc_real = P[idx]

        axs[i][0].imshow(X[idx])
        axs[i][0].set_title('{}'.format(idx_word_dict[Y[idx]]))

        draw_bounding_box(axs[i][0], bbox_list[idx], color='green')
        draw_bounding_box(axs[i][0], loc_real, color='red')
        axs[i][1].imshow(X[idx])
        axs[i][1].imshow(cam_list[idx],
                         cmap=plt.cm.jet,
                         alpha=0.5,
                         interpolation='nearest')
        draw_bounding_box(axs[i][1], bbox_list[idx], color='green')
        draw_bounding_box(axs[i][1], loc_real, color='red')
    return fig, axs

def visualize_cam(X, Y, cam_list, idx_word_dict, n_show=3, start=0):
    fig, axs = plt.subplots(n_show, 2, figsize=(6, 6*n_show))
    for i in range(n_show):
        idx = start + i 

        axs[i][0].imshow(X[idx])
        axs[i][0].set_title('{}'.format(idx_word_dict[Y[idx]]))

        axs[i][1].imshow(X[idx])
        axs[i][1].imshow(cam_list[idx],
                         cmap=plt.cm.jet,
                         alpha=0.5,
                         interpolation='nearest')
    return fig, axs

def restore_preprocessed(X):
    result = (X*0.5+0.5)*255
    return result.astype(np.uint8)
