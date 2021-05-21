import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0], '../'))
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
from core.concentration import *

HIT_COLOR = np.array([255, 255, 255])
MISSED_COLOR = np.array([255, 69, 85])
MISFIRE_COLOR = np.array([64, 181, 188])

def plot_grid(list_img_list, list_result_list, output_dir):
    fig, axs = plt.subplots(nrows = 2*len(list_result_list), ncols = len(list_img_list[0]), figsize = (len(list_img_list[0])*10,10*2*len(list_result_list)))
    for i in range(len(list_result_list)):
        for j in range(len(list_result_list[0])):
            axs[2*i,j].axis('off')
            axs[2*i,j].imshow(list_img_list[i][j], aspect='equal')
            axs[2*i+1,j].axis('off')
            axs[2*i+1,j].imshow(list_result_list[i][j], aspect='equal')
    plt.tight_layout()
    plt.savefig(output_dir + 'conformal_grid_fig.pdf')

def get_results(lhat, nc_list, val_img_names, val_scores, val_masks, val_num_components, num_plot):
    list_img_list = list()
    list_result_list = list()
    for i in range(len(nc_list)):
        nc = nc_list[i]
        filter_bool = val_num_components == nc 
        val_img_names_nc = val_img_names[filter_bool]
        val_scores_nc = val_scores[filter_bool]
        val_masks_nc = val_masks[filter_bool]

        Tlamhat = val_scores_nc[0:num_plot] >= -lhat 

        val_masks_nc = (val_masks_nc > 0).to(float)
        result = val_masks_nc[0:num_plot]
        result[result == 0] = -2
        result = result - Tlamhat.to(float)

        img_list = list()
        result_list = list()
        for i in range(num_plot):
            res = result[i]
            result_display = np.zeros((res.shape[0], res.shape[1], 3))
            result_display[res == 0] = HIT_COLOR/255.
            result_display[res == -3] = MISFIRE_COLOR/255.
            result_display[res == 1] = MISSED_COLOR/255.

            result_list = result_list + [result_display]

            img = io.imread(val_img_names_nc[i])
            img_list = img_list + [resize(img, (result_display.shape[0], result_display.shape[1]))]
        list_img_list = list_img_list + [img_list]
        list_result_list = list_result_list + [result_list]

    return list_img_list, list_result_list


def get_grid(gamma, delta, num_plot, num_calib, num_lam, deltas_precomputed, num_grid_hbb, ub, ub_sigma, epsilon, maxiters, output_dir):
    img_names, sigmoids, masks, regions, num_components = get_data(cache_path)

    calib_img_names, val_img_names, calib_sigmoids, val_sigmoids, calib_masks, val_masks, calib_regions, val_regions, calib_num_components, val_num_components = calib_test_split((img_names, sigmoids, masks, regions, num_components), num_calib)
    # Calibrate
    lambda_hat_conformal = get_lambda_hat_conformal(calib_regions, calib_masks, gamma, delta)
    nc_list = [1,2]

    list_img_list, list_result_list = get_results(lambda_hat_conformal, nc_list, val_img_names, val_regions, val_masks, val_num_components, num_plot)
    
    return list_img_list, list_result_list

if __name__ == '__main__':
    with torch.no_grad():
        sns.set(palette='pastel', font='serif')
        sns.set_style('white')
        fix_randomness(seed=5)

        cache_path = './.cache/'
        output_dir = 'outputs/grid_fig/'
        pathlib.Path(cache_path).mkdir(parents=True, exist_ok=True)
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

        num_plot = 10 
        num_calib = 1000 
        num_grid_hbb = 200
        epsilon = 1e-10
        maxiters = int(1e5)
        ub = 0.2
        ub_sigma = np.sqrt(2)
        num_lam = 1500
        #lam_lim = [-0.8,-0.30]
        gamma = 0.1
        delta = 0.1
        deltas_precomputed = [0.001, 0.01, 0.05, 0.1]
        list_img_list, list_result_list = get_grid(gamma, delta, num_plot, num_calib, num_lam, deltas_precomputed, num_grid_hbb, ub, ub_sigma, epsilon, maxiters, output_dir)
        plot_grid(list_img_list, list_result_list, output_dir)
