# coding: utf-8

from cv2 import cv2
import glob
import os


def imread(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    return img[:, :, 0]


def prepare_data(path):
    data_path = os.path.join(path)
    images_path = glob.glob(os.path.join(data_path, "*.bmp"))
    images_path.extend(glob.glob(os.path.join(data_path, "*.tif")))
    images_path.sort(key=lambda x: int(x[len(data_path) + len(os.path.sep):-4]))
    return images_path
