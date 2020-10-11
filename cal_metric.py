# coding: utf-8

import numpy as np
import pandas as pd
import argparse
import os
from skimage.measure import shannon_entropy

from utils import prepare_data, imread


def EN(img):
    return shannon_entropy(img)


def SD(img):
    return np.std(img)


def cross_covariance(x, y, mu_x, mu_y):
    return 1 / (x.size - 1) * np.sum((x - mu_x) * (y - mu_y))


def SSIM(x, y):
    L = np.max(np.array([x, y])) - np.min(np.array([x, y]))
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sig_x = np.std(x)
    sig_y = np.std(y)
    sig_xy = cross_covariance(x, y, mu_x, mu_y)
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2
    C3 = C2 / 2
    return (2 * mu_x * mu_y + C1) * (2 * sig_x * sig_y + C2) * (sig_xy + C3) / (
            (mu_x ** 2 + mu_y ** 2 + C1) * (sig_x ** 2 + sig_y ** 2 + C2) * (sig_x * sig_y + C3))


def correlation_coefficients(x, y):
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    return np.sum((x - mu_x) * (y - mu_y)) / np.sqrt(np.sum((x - mu_x) ** 2) * np.sum((y - mu_y) ** 2))


def CC(ir, vi, fu):
    rx = correlation_coefficients(ir, fu)
    ry = correlation_coefficients(vi, fu)
    return (rx + ry) / 2


def SF(I):
    I = I.astype(np.int16)
    RF = np.diff(I, 1, 0)
    # RF[RF < 0] = 0
    RF = RF ** 2
    # RF[RF > 255] = 255
    RF = np.sqrt(np.mean(RF))

    CF = np.diff(I, 1, 1)
    # CF[CF < 0] = 0
    CF = CF ** 2
    # CF[CF > 255] = 255
    CF = np.sqrt(np.mean(CF))
    return np.sqrt(RF ** 2 + CF ** 2)


parser = argparse.ArgumentParser(description='infrared visible fusion metrics')
parser.add_argument('--data_inf_path', help='folder for infrared images', required=True)
parser.add_argument('--data_vis_path', help='folder for visible images', required=True)
parser.add_argument('--data_fus_path', help='folder for fusion images', required=True)
parser.add_argument('--fusion_model', help='fusion model', required=True)
args = parser.parse_args()

if __name__ == '__main__':
    data_inf_path = args.data_inf_path
    data_vis_path = args.data_vis_path
    data_fus_path = args.data_fus_path

    data_inf = prepare_data(data_inf_path)
    data_vis = prepare_data(data_vis_path)
    data_fus = prepare_data(data_fus_path)

    titles = ['EN', 'SD', 'SSIM', 'CC', 'SF']
    metrics = []
    for i in range(len(data_inf)):
        inf = imread(data_inf[i])
        vis = imread(data_vis[i])
        fus = imread(data_fus[i])

        en = EN(fus)
        sd = SD(fus)
        ssim = SSIM(inf, fus) + SSIM(vis, fus)
        cc = CC(inf, vis, fus)
        sf = SF(fus)

        metrics.append([en, sd, ssim, cc, sf])
    metrics = np.array(metrics)
    data = pd.DataFrame(metrics, columns=titles)
    try:
        result_folder = 'result'
        if not os.path.exists(result_folder):
            os.mkdir(result_folder)
        save_path = os.path.join(result_folder, args.fusion_model + '.csv')
        pd.DataFrame.to_csv(data, save_path)
    except Exception as e:
        print('Error when saving result.', e)
    else:
        print('Saving result to ' + save_path)
