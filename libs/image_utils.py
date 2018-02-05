import numpy as np

import matplotlib.patches as mpatches

from skimage.measure import label, regionprops
from skimage.morphology import closing, square


def get_random_patch_list(img_size, patch_size):
    if img_size % patch_size != 0:
        raise Exception("patch_size cannot divide by img_size")
    patch_num = img_size//patch_size
    patch_list = [(x*patch_size, y*patch_size, patch_size, patch_size)
                  for y in range(patch_num)
                  for x in range(patch_num)]
    return patch_list


def random_hide(img, patch_list, hide_prob=0.5, mean=0.5):
    if type(img) is not np.ndarray:
        img = np.array(img)
    img = img.copy()
    np.random.seed()
    for patch in patch_list:
        (x, y, width, height) = patch
        if np.random.uniform() < hide_prob:
            img[x:x+width, y:y+height] = mean
    return img


def find_biggest_bbox(binary_img):
    label_list = label(binary_img)

    bbox = None
    max_area = 0
    for region in regionprops(label_list):
        if region.area > max_area:
            minr, minc, maxr, maxc = region.bbox
            bbox = (minc, minr, maxc - minc, maxr - minr)
            max_area = region.area
    return bbox


def draw_bounding_box(ax,
                      bbox,
                      fill=False,
                      color='red',
                      linewidth=2):
    x, y, width, height = bbox
    rect = mpatches.Rectangle(
        (x, y), width, height,
        fill=fill, color=color, linewidth=linewidth)
    ax.add_patch(rect)


def binarize(img, thresh):
    return closing(img > thresh, square(3))


def find_location_by_cam(cam, thresh):
    cam = (cam - np.min(cam))/(np.max(cam)-np.min(cam))
    binary_img = binarize(cam, thresh)
    bbox = find_biggest_bbox(binary_img) 
    return bbox


def find_intersaction(bbox_1, bbox_2):
    x_left = max(bbox_1[0], bbox_2[0])
    y_top = max(bbox_1[1], bbox_2[1])
    x_right = min(bbox_1[0] + bbox_1[2], bbox_2[0] + bbox_2[2])
    y_bottom = min(bbox_1[1] + bbox_1[3], bbox_2[1] + bbox_2[3])
    return (x_left, y_top,  x_right-x_left, y_bottom-y_top)
 

def calc_iou(bbox_1, bbox_2):
    bbox_inter = find_intersaction(bbox_1, bbox_2)
    bbox_inter_size = (bbox_inter[2] + 1) * (bbox_inter[3] + 1)
    bbox_1_size = (bbox_1[2]+1) * (bbox_1[3]+1)
    bbox_2_size = (bbox_2[2]+1) * (bbox_2[3]+1)
    iou = bbox_inter_size / float(bbox_1_size + bbox_2_size - bbox_inter_size)
    return iou


def calc_iou_accuracy(bbox_pred_list, bbox_real_list, thresh_iou=0.5):
    correct = 0

    for i, bbox_pred in enumerate(bbox_pred_list):
        bbox_real = bbox_real_list[i]
        iou = calc_iou(bbox_real, bbox_pred)
        if iou > thresh_iou:
            correct += 1
    accuracy = correct/len(bbox_pred_list)
    return accuracy


def calc_iou_top_1_accuracy(bbox_pred_list,
                            Y_pred,
                            bbox_real_list,
                            Y_real,
                            thresh_iou=0.5):
    correct = 0

    for i, bbox_pred in enumerate(bbox_pred_list):
        if not np.array_equal(Y_pred[i], Y_real[i]):
            continue

        bbox_real = bbox_real_list[i]
        iou = calc_iou(bbox_real, bbox_pred)
        if iou > thresh_iou:
            correct += 1
    accuracy = correct/len(bbox_pred_list)
    return accuracy
        

