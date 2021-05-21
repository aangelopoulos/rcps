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

def get_example_loss_and_size_tables(regions, masks, lambdas_example_table, num_calib):
    lam_len = len(lambdas_example_table)
    lam_low = min(lambdas_example_table)
    lam_high = max(lambdas_example_table)
    fname_loss = f'.cache/{lam_low}_{lam_high}_{lam_len}_example_loss_table.npy'
    fname_sizes = f'.cache/{lam_low}_{lam_high}_{lam_len}_example_size_table.npy'
    try:
        loss_table = np.load(fname_loss)
        sizes_table = np.load(fname_sizes)
    except:
        print("computing loss and size table")
        loss_table = np.zeros((regions.shape[0], lam_len))
        sizes_table = np.zeros((regions.shape[0], lam_len))
        for j in tqdm(range(lam_len)):
            est_regions = (regions >= -lambdas_example_table[j])
            loss_table[:,j] = loss_perpolyp_01(est_regions, regions, masks) 
            sizes_table[:,j] = est_regions.sum(dim=1).sum(dim=1)/masks.sum(dim=1).sum(dim=1)

        np.save(fname_loss, loss_table)
        np.save(fname_sizes, sizes_table)

    return loss_table, sizes_table

def trial_precomputed(example_loss_table, example_size_table,  gamma, delta, num_calib, num_lam, deltas_precomputed, num_grid_hbb, ub, ub_sigma, lambdas_example_table, epsilon, maxiters, tlambda, bound_str):
    total=example_loss_table.shape[0]
    perm = torch.randperm(example_loss_table.shape[0])
    example_loss_table = example_loss_table[perm]
    example_size_table = example_size_table[perm]
    calib_losses, val_losses = (example_loss_table[0:num_calib], example_loss_table[num_calib:])
    calib_sizes, val_sizes = (example_size_table[0:num_calib], example_size_table[num_calib:])

    lhat_rcps = get_lhat_from_table(calib_losses, lambdas_example_table, gamma, delta, tlambda, bound_str)

    losses_rcps = val_losses[:,np.argmax(lambdas_example_table == lhat_rcps)]
    sizes_rcps = val_sizes[:,np.argmax(lambdas_example_table == lhat_rcps)]

    return lhat_rcps, losses_rcps.mean(), sizes_rcps

def plot_histograms(dfs, gamma, delta, num_calib, output_dir):
    fig = plt.figure(figsize=(6,3))
    ax = plt.axes()
    for df in dfs:
        plt.hist(np.array(df['risk'].tolist()), alpha=0.7, density=True)

        plt.xlabel('risk')
        ax.locator_params(axis='x', nbins=5)
        ax.axvline(x=gamma,c='#999999',linestyle='--',alpha=0.7)
        plt.ylabel('density')
    sns.despine(top=True, right=True, ax=ax)
    plt.tight_layout()
    plt.savefig( output_dir + (f'simplified_{gamma}_{delta}_{num_calib}_polyp_clt_wsr_histograms').replace('.','_') + '.pdf'  )

def experiment(gamma, delta, num_trials, num_calib, num_lam, output_dir, bound, deltas_precomputed, num_grid_hbb, ub, ub_sigma, epsilon, maxiters, lambdas_example_table):
    img_names, sigmoids, masks, regions, num_components = get_data(cache_path)
    masks[masks > 1] = 1
    fname = cache_path + f'{gamma}_{delta}_{num_calib}_{num_lam}_{bound}_dataframe'.replace('.','_') + '.pkl'

    df = pd.DataFrame(columns=['bound','$\\hat{\\lambda}$','risk','sizes','gamma','delta'])
    try:
        print('Dataframe loaded')
        df = pd.read_pickle(fname)
    except:
        print('Performing experiments from scratch.')
        if bound == 'Bentkus':
            bound = bentkus_mu_plus
        elif bound == 'HB':
            bound_fn = HB_mu_plus
        elif bound == 'HBB':
            bound_fn = HBB_mu_plus
        elif bound == 'WSR':
            bound_fn = WSR_mu_plus
        elif bound == 'CLT':
            bound_fn = None 
        else:
            raise NotImplemented

        tlambda = get_tlambda(num_lam,deltas_precomputed,num_calib,num_grid_hbb,ub,ub_sigma,epsilon,maxiters,bound.lower(),bound_fn)
        example_loss_table, example_sizes_table = get_example_loss_and_size_tables(regions, masks, lambdas_example_table, num_calib)

        local_df_list = []
        for i in tqdm(range(num_trials)):
            lhat, risk, sizes = trial_precomputed(example_loss_table, example_sizes_table,  gamma, delta, num_calib, num_lam, deltas_precomputed, num_grid_hbb, ub, ub_sigma, lambdas_example_table, epsilon, maxiters, tlambda, bound)
                
            dict_local = {  "bound": bound, 
                            "$\\hat{\\lambda}$": lhat,
                            "risk": risk,
                            "sizes": [sizes],
                            "gamma": gamma,
                            "delta": delta
                         }
            df_local = pd.DataFrame(dict_local)
            local_df_list = local_df_list + [df_local]
        df = pd.concat(local_df_list, axis=0, ignore_index=True)
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

        num_trials = 1000 
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
        bounds = ['WSR']
        lambdas_example_table = np.linspace(-1,0,1000)

        dfs = []
        for bound in bounds:
            dfs = dfs + [experiment(gamma, delta, num_trials, num_calib, num_lam, output_dir, bound, deltas_precomputed, num_grid_hbb, ub, ub_sigma, epsilon, maxiters, lambdas_example_table)]
        plot_histograms(dfs, gamma, delta, num_calib, output_dir)
