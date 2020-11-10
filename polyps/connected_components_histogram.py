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

# TODO: All of this is very preliminary code for the CLT only.  Will need to expand (see imagenet)
def plot_histogram(num_components, output_dir):
    plt.hist(num_components, alpha=0.7, density=True)
    ax = plt.gca()
    sns.despine(top=True, right=True, ax=ax)
    plt.tight_layout()
    plt.savefig( output_dir + 'num_connected_components_histogram.pdf' )

def plot_examples(img_names, sigmoids, masks, num_components, desired_num_components, num_images):
    fig, axs = plt.subplots(nrows = 3*len(desired_num_components), ncols = num_images, figsize = (num_images * 10, 10*(3*len(desired_num_components)))) 
    for i in range(num_images):
        for r in range(len(desired_num_components)):
            pdb.set_trace()
            filtered_names = img_names[num_components == desired_num_components[r]]
            filtered_sigmoids = sigmoids[num_components == desired_num_components[r]]
            filtered_masks = masks[num_components == desired_num_components[r]]
            if filtered_masks.shape[0] <= i:
                continue
            axs[3*r,i].axis('off')
            axs[3*r,i].imshow(io.imread(filtered_names[i]), aspect='equal')
            axs[3*r+1,i].axis('off')
            axs[3*r+1,i].imshow(find_peaks(filtered_sigmoids[i]), aspect='equal')
            axs[3*r+2,i].axis('off')
            axs[3*r+2,i].imshow(filtered_masks[i], aspect='equal')
    plt.tight_layout()
    plt.savefig(f'outputs/grid_fig/{desired_num_components}_conn_comp_grid_fig.pdf')

if __name__ == '__main__':
    sns.set(palette='pastel', font='serif')
    sns.set_style('white')
    fix_randomness()

    cache_path = './.cache/'
    output_dir = 'outputs/histograms/'
    pathlib.Path(cache_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    img_names, sigmoids, masks, regions, num_components = get_data(cache_path)
    plot_histogram(num_components, output_dir)
    plot_examples(img_names, sigmoids, masks, num_components, (1,2,3), 5) 
