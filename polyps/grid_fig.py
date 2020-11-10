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

def plot_grid(img_list, result_01_list, result_adaptive_list, output_dir):
    fig, axs = plt.subplots(nrows = 3, ncols = len(img_list), figsize = (len(img_list)*10,30))
    for i in range(len(img_list)):
        axs[0,i].axis('off')
        axs[0,i].imshow(img_list[i], aspect='equal')
        axs[1,i].axis('off')
        axs[1,i].imshow(result_01_list[i], aspect='equal')
        axs[2,i].axis('off')
        axs[2,i].imshow(result_adaptive_list[i], aspect='equal')
    plt.tight_layout()
    plt.savefig(output_dir + 'polyp_grid_fig.pdf')

def get_grid(gamma, delta, num_plot, num_calib, num_lam, lam_lim, output_dir):
    img_names, sigmoids, masks, regions, num_components = get_data(cache_path)
    #masks[masks > 1] = 1 # TODO: EVENTUALLY REMOVE THIS AND DEAL WITH CONNECTED COMPONENTS SEPARATELY.
    calib_img_names, val_img_names, calib_sigmoids, val_sigmoids, calib_masks, val_masks, calib_regions, val_regions, calib_num_components, val_num_components = calib_test_split((img_names, sigmoids, masks, regions, num_components), num_calib)
    # Calibrate
    lambda_hat_01 = get_lambda_hat_01(calib_sigmoids, (calib_masks > 0).int(), gamma, delta, num_lam, lam_lim)
    
    lambda_hat_adaptive = get_lambda_hat_perpolyp_01(calib_regions, calib_masks, gamma, delta, num_lam, lam_lim)

    # Construct sets TODO: make this more general with num_components
    filter_bool = val_num_components == 2 
    val_img_names = val_img_names[filter_bool]
    val_sigmoids = val_sigmoids[filter_bool]
    val_masks = val_masks[filter_bool]
    val_regions = val_regions[filter_bool]

    Tlamhat_01 = val_sigmoids[0:num_plot] >= -lambda_hat_01 

    Tlamhat_adaptive = val_regions[0:num_plot] >= -lambda_hat_adaptive #construct_set_l2(val_sigmoids[0:num_plot], 0.01, lambda_hat_l2)
    #efficiency = masks.to(float).mean()/Tlamhat.to(float).mean()
    #print(f"lambda {lambda_hat:.3f} has efficiency {efficiency:.3f}")
    val_masks = (val_masks > 0).to(float)
    result_01 = val_masks[0:num_plot]
    result_01[result_01 == 0] = -2
    result_01 = result_01 - Tlamhat_01.to(float)

    result_adaptive = val_masks[0:num_plot]
    result_adaptive[result_adaptive == 0] = -2
    result_adaptive = result_adaptive - Tlamhat_adaptive.to(float)
    img_list = list()
    result_01_list = list()
    result_adaptive_list = list()
    for i in range(num_plot):
        res_01 = result_01[i]
        result_01_display = np.zeros((res_01.shape[0], res_01.shape[1], 3))
        result_01_display[res_01 == 0] = HIT_COLOR/255.
        result_01_display[res_01 == -3] = MISFIRE_COLOR/255.
        result_01_display[res_01 == 1] = MISSED_COLOR/255.

        result_01_list = result_01_list + [result_01_display]

        res_adaptive = result_adaptive[i]
        result_adaptive_display = np.zeros((res_adaptive.shape[0], res_adaptive.shape[1], 3))
        result_adaptive_display[res_adaptive == 0] = HIT_COLOR/255.
        result_adaptive_display[res_adaptive == -3] = MISFIRE_COLOR/255.
        result_adaptive_display[res_adaptive == 1] = MISSED_COLOR/255.

        result_adaptive_list = result_adaptive_list + [result_adaptive_display]
        
        img = io.imread(val_img_names[i])
        img_list = img_list + [resize(img, (result_01_display.shape[0], result_01_display.shape[1]))]

    return img_list, result_01_list, result_adaptive_list

if __name__ == '__main__':
    with torch.no_grad():
        sns.set(palette='pastel', font='serif')
        sns.set_style('white')
        fix_randomness(seed=5)

        cache_path = './.cache/'
        output_dir = 'outputs/grid_fig/'
        pathlib.Path(cache_path).mkdir(parents=True, exist_ok=True)
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

        num_plot = 15 
        num_calib = 1000 
        num_lam = 100 
        lam_lim = [-0.8,-0.30]
        gamma = 0.2
        delta = 0.2

        img_list, result_01_list, result_adaptive_list = get_grid(gamma, delta, num_plot, num_calib, num_lam, lam_lim, output_dir)
        plot_grid(img_list, result_01_list, result_adaptive_list, output_dir)
