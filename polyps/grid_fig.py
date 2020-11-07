import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
import imageio as io
import matplotlib.pyplot as plt
import pandas as pd
from polyp_utils import *
from PraNet.lib.PraNet_Res2Net import PraNet
from PraNet.utils.dataloader import test_dataset
import pathlib
import random
from scipy.stats import norm
from skimage.transform import resize
import seaborn as sns
from tqdm import tqdm
import pdb

HIT_COLOR = np.array([255, 255, 255])
MISSED_COLOR = np.array([255, 69, 85])
MISFIRE_COLOR = np.array([64, 181, 188])

def plot_grid(img_list, result_list,output_dir):
    fig, axs = plt.subplots(nrows = 2, ncols = len(img_list), figsize = (len(img_list)*10,20))
    for i in range(len(img_list)):
        axs[0,i].axis('off')
        axs[0,i].imshow(img_list[i], aspect='equal')
        axs[1,i].axis('off')
        axs[1,i].imshow(result_list[i], aspect='equal')
    plt.tight_layout()
    plt.savefig(output_dir + 'polyp_grid_fig.pdf')

def get_grid(gamma, delta, num_plot, num_calib, num_lam, lam_lim, output_dir):
    img_names, sigmoids, masks = get_data(cache_path)
    calib_img_names, val_img_names, calib_sigmoids, calib_masks, val_sigmoids, val_masks = calib_test_split(img_names, sigmoids, masks, num_calib)
    lambda_hat = get_lambda_hat_l2(calib_sigmoids, calib_masks, gamma, delta, num_lam, lam_lim)
    efficiency = masks.to(float).mean()/(val_sigmoids >= -lambda_hat).to(float).mean()
    print(f"lambda {lambda_hat:.3f} has efficiency {efficiency:.3f}")
    result = val_masks
    result[result == 0] = -2
    result = result - (val_sigmoids >= -lambda_hat).to(float)
    img_list = list()
    result_list = list()
    for i in range(num_plot):
        res = result[i]
        result_display = np.zeros((res.shape[0], res.shape[1], 3))
        result_display[res == 0] = HIT_COLOR/255.
        result_display[res == -3] = MISFIRE_COLOR/255.
        result_display[res == 1] = MISSED_COLOR/255.

        result_list = result_list + [result_display]
        
        img = io.imread(val_img_names[i])
        img_list = img_list + [resize(img, (result_display.shape[0], result_display.shape[1]))]

    return img_list, result_list

if __name__ == '__main__':
    with torch.no_grad():
        sns.set(palette='pastel', font='serif')
        sns.set_style('white')
        fix_randomness(seed=5)

        cache_path = './.cache/'
        output_dir = 'outputs/grid_fig/'
        pathlib.Path(cache_path).mkdir(parents=True, exist_ok=True)
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

        num_plot = 5 
        num_calib = 1000
        num_lam = 10
        lam_lim = [-0.4, -0.3]
        gamma = 0.1
        delta = 0.1

        img_list, result_list = get_grid(gamma, delta, num_plot, num_calib, num_lam, lam_lim, output_dir)
        plot_grid(img_list, result_list, output_dir)
