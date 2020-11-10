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

def trial(img_names, sigmoids, masks, num_components, gamma, delta, num_calib, num_lam, lam_lim):
    _, _, calib_sigmoids, val_sigmoids, calib_masks, val_masks = calib_test_split(img_names, sigmoids, masks, num_components, num_calib)
    lambda_hat = get_lambda_hat(calib_sigmoids, calib_masks, gamma, delta, risk_01, num_lam, lam_lim)
    empirical_risk, _ = risk_01(val_sigmoids, val_masks, lambda_hat)
    avg_polyp_size = val_masks.sum(dim=1).sum(dim=1).mean()
    avg_set_size = (val_sigmoids >= -lambda_hat).float().sum(dim=1).sum(dim=1).mean()
    return lambda_hat, empirical_risk, avg_set_size, avg_polyp_size

# TODO: All of this is very preliminary code for the CLT only.  Will need to expand (see imagenet)
def plot_histogram(df, gamma, delta, num_calib, output_dir):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12,3))

    axs[0].hist(np.array(df['risk'].tolist()), alpha=0.7, density=True)

    normalized_size = np.array(df['average size'].tolist())/np.array(df['average polyp size'].tolist()).mean() 
    axs[1].hist(normalized_size, alpha=0.7, density=True)

    axs[0].set_xlabel('risk')
    axs[0].locator_params(axis='x', nbins=5)
    axs[0].set_ylabel('density')
    axs[1].set_xlabel('set size as a fraction of mean polyp size')
    axs[1].locator_params(axis='x', nbins=5)
    sns.despine(top=True, right=True, ax=axs[0])
    sns.despine(top=True, right=True, ax=axs[1])
    plt.tight_layout()
    plt.savefig( output_dir + (f'{gamma}_{delta}_{num_calib}_polyp_l2_histograms').replace('.','_') + '.pdf'  )

def experiment(gamma, delta, num_trials, num_calib, num_lam, lam_lim, output_dir):
    fname = cache_path + f'{gamma}_{delta}_{num_calib}_{num_lam}_dataframe'.replace('.','_') + '.pkl'
    img_names, sigmoids, masks, num_components = get_data(cache_path)

    df = pd.DataFrame(columns=['$\\hat{\\lambda}$','risk','average size', 'average polyp size','gamma','delta'])
    try:
        print('Dataframe loaded')
        df = pd.read_pickle(fname)
    except:
        print('Performing experiments from scratch.')
        for i in tqdm(range(num_trials)):
            lambda_hat, empirical_risk, avg_set_size, avg_polyp_size = trial(img_names, sigmoids, masks, num_components, gamma, delta, num_calib, num_lam, lam_lim) 
            df = df.append({'$\\hat{\\lambda}$': lambda_hat,
                            'risk': empirical_risk,
                            'average size': avg_set_size,
                            'average polyp size': avg_polyp_size,
                            'gamma': gamma,
                            'delta': delta}, ignore_index=True)
        df.to_pickle(fname)
    plot_histogram(df, gamma, delta, num_calib, output_dir)

if __name__ == '__main__':
    with torch.no_grad():
        sns.set(palette='pastel', font='serif')
        sns.set_style('white')
        fix_randomness()

        cache_path = './.cache/'
        output_dir = 'outputs/histograms/'
        pathlib.Path(cache_path).mkdir(parents=True, exist_ok=True)
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

        num_trials = 100
        num_calib = 1000
        num_lam = 10
        lam_lim = [-0.5, -0.4]
        gamma = 0.1
        delta = 0.1

        experiment(gamma, delta, num_trials, num_calib, num_lam, lam_lim, output_dir)
