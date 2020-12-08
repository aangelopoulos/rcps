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

def trial_clt(img_names, regions, masks, num_components, gamma, delta, num_calib, num_lam):
    _, _, calib_regions, val_regions, calib_masks, val_masks, calib_num_components, val_num_components = calib_test_split((img_names, regions, masks, num_components), num_calib)
    lambda_hat = get_lambda_hat_clt_perpolyp_01(calib_regions, calib_masks, gamma, delta, num_lam)
    est_regions = (val_regions >= -lambda_hat)
    empirical_risk, _ = empirical_risk_perpolyp_01(est_regions, val_regions, val_masks)
    avg_polyp_size = val_masks.sum(dim=1).sum(dim=1).mean()
    avg_set_size = est_regions.float().sum(dim=1).sum(dim=1).mean()
    return lambda_hat, empirical_risk, avg_set_size, avg_polyp_size

def trial_hbb(img_names, regions, masks, num_components, gamma, delta, num_calib, num_lam, deltas_precomputed, num_grid_hbb, ub, ub_sigma, epsilon, maxiters):
    _, _, calib_regions, val_regions, calib_masks, val_masks, calib_num_components, val_num_components = calib_test_split((img_names, regions, masks, num_components), num_calib)
    tlambda = get_tlambda(num_lam, deltas_precomputed, num_calib, num_grid_hbb, ub, ub_sigma, epsilon, maxiters, 'HBB', HBB_mu_plus)
    lambda_hat = get_lambda_hat_hbb_perpolyp_01(calib_regions, calib_masks, gamma, delta, num_lam, tlambda)
    est_regions = (val_regions >= -lambda_hat)
    empirical_risk, _ = empirical_risk_perpolyp_01(est_regions, val_regions, val_masks)
    avg_polyp_size = val_masks.sum(dim=1).sum(dim=1).mean()
    avg_set_size = est_regions.float().sum(dim=1).sum(dim=1).mean()
    return lambda_hat, empirical_risk, avg_set_size, avg_polyp_size

def plot_histograms(dfs, gamma, delta, num_calib, output_dir):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12,3))
    for df in dfs:
        axs[0].hist(np.array(df['risk'].tolist()), alpha=0.7, density=True)

        normalized_size = np.array(df['average size'].tolist())/np.array(df['average polyp size'].tolist()).mean() 
        axs[1].hist(normalized_size, alpha=0.7, label=df.bound[0].upper(), density=True)

        axs[0].set_xlabel('risk')
        axs[0].locator_params(axis='x', nbins=5)
        axs[0].set_ylabel('density')
        axs[1].set_xlabel('set size as a fraction of mean polyp size')
        axs[1].locator_params(axis='x', nbins=5)
        axs[1].legend()
    sns.despine(top=True, right=True, ax=axs[0])
    sns.despine(top=True, right=True, ax=axs[1])
    plt.tight_layout()
    plt.savefig( output_dir + (f'{gamma}_{delta}_{num_calib}_polyp_clt_hbb_histograms').replace('.','_') + '.pdf'  )

def experiment(gamma, delta, num_trials, num_calib, num_lam, output_dir,bound, deltas_precomputed, num_grid_hbb, ub, ub_sigma, epsilon, maxiters):
    fname = cache_path + f'{gamma}_{delta}_{num_calib}_{num_lam}_{bound}_dataframe'.replace('.','_') + '.pkl'
    img_names, sigmoids, masks, regions, num_components = get_data(cache_path)
    masks[masks > 1] = 1

    df = pd.DataFrame(columns=['bound','$\\hat{\\lambda}$','risk','average size', 'average polyp size','gamma','delta'])
    try:
        print('Dataframe loaded')
        df = pd.read_pickle(fname)
    except:
        print('Performing experiments from scratch.')
        for i in tqdm(range(num_trials)):
            if bound == 'clt':
                lambda_hat, empirical_risk, avg_set_size, avg_polyp_size = trial_clt(img_names, regions, masks, num_components, gamma, delta, num_calib, num_lam) 
            elif bound == 'hbb':
                lambda_hat, empirical_risk, avg_set_size, avg_polyp_size = trial_hbb(img_names, regions, masks, num_components, gamma, delta, num_calib, num_lam, deltas_precomputed, num_grid_hbb, ub, ub_sigma, epsilon, maxiters) 
                
            df = df.append({'bound': bound,
                            '$\\hat{\\lambda}$': lambda_hat,
                            'risk': empirical_risk,
                            'average size': avg_set_size,
                            'average polyp size': avg_polyp_size,
                            'gamma': gamma,
                            'delta': delta}, ignore_index=True)
        df.to_pickle(fname)
    return df

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
        num_lam = 1500
        gamma = 0.1
        delta = 0.1
        ub = 0.2
        ub_sigma = np.sqrt(2)
        epsilon = 1e-10
        maxiters = int(1e5)
        num_grid_hbb = 200
        deltas_precomputed = [0.001, 0.01, 0.05, 0.1]
        bounds = ['clt','hbb']

        dfs = []
        for bound in bounds:
            dfs = dfs + [experiment(gamma, delta, num_trials, num_calib, num_lam, output_dir, bound, deltas_precomputed, num_grid_hbb, ub, ub_sigma, epsilon, maxiters)]
        plot_histograms(dfs, gamma, delta, num_calib, output_dir)
