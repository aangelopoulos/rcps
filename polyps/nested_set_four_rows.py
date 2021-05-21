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

def plot_grid(fig, axs, list_list_img_list, list_list_result_list):
    for j in range(len(list_list_result_list[0][0])):
        axs[0,j].imshow(list_list_img_list[0][0][j], aspect='equal')
        axs[0,j].axis('off')
        for i in range(1,4):
            axs[i,j].imshow(list_list_result_list[i-1][0][j], aspect='equal')
            axs[i,j].axis('off')
            if j == 0:
                axs[i,j].axis('on')
                axs[i,j].xaxis.set_ticks([])
                axs[i,j].yaxis.set_ticks([])
                axs[i,j].xaxis.set_ticklabels([])
                axs[i,j].yaxis.set_ticklabels([])

    axs[0,0].axis('on')
    axs[0,0].xaxis.set_ticks([])
    axs[0,0].yaxis.set_ticks([])
    axs[0,0].xaxis.set_ticklabels([])
    axs[0,0].yaxis.set_ticklabels([])
    axs[0,0].set_ylabel('polyps', fontsize = 90)
    axs[1,0].set_ylabel(r'$\lambda=-0.99$', fontsize = 90)
    axs[2,0].set_ylabel(r'$\lambda=-0.33$', fontsize = 90)
    axs[3,0].set_ylabel(r'$\lambda=-0.15$', fontsize = 90)
    fig.suptitle(r'Nested sets at different values of $\lambda$', fontsize = 120)
    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])

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


def save_figure(gamma, delta, num_plot, num_calib, num_lam, deltas_precomputed, num_grid_hbb, ub, ub_sigma, epsilon, maxiters, output_dir):
    img_names, sigmoids, masks, regions, num_components = get_data(cache_path)

    calib_img_names, val_img_names, calib_sigmoids, val_sigmoids, calib_masks, val_masks, calib_regions, val_regions, calib_num_components, val_num_components = calib_test_split((img_names, sigmoids, masks, regions, num_components), num_calib)
    # Calibrate
    nc_list = [1]

    fig, axs = plt.subplots(nrows = 4, ncols = 10, figsize = (88,40))
    idx = 0
    lams = [-0.99, -0.33, -0.15]
    list_list_img_list = []
    list_list_result_list = []
    for lam in lams:
        list_img_list, list_result_list = get_results(lam, nc_list, val_img_names, val_regions, val_masks, val_num_components, num_plot)
        list_list_img_list = list_list_img_list + [list_img_list]
        list_list_result_list = list_list_result_list + [list_result_list]
    plot_grid(fig, axs, list_list_img_list, list_list_result_list)
    plt.savefig(output_dir + 'keynote_fig.png') 

if __name__ == '__main__':
    with torch.no_grad():
        sns.set(palette='pastel', font='serif')
        sns.set_style('white')
        fix_randomness(seed=5)

        cache_path = './.cache/'
        output_dir = 'outputs/nestedsets_keynote/'
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
        save_figure(gamma, delta, num_plot, num_calib, num_lam, deltas_precomputed, num_grid_hbb, ub, ub_sigma, epsilon, maxiters, output_dir)
