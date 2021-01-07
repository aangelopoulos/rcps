import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0], '../'))
import torch
import torchvision as tv
import argparse
import time
import numpy as np
from scipy.stats import binom
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pkl
from tqdm import tqdm
import seaborn as sns
from core.concentration import *
from scipy.optimize import brentq
from utils import *
from protein_utils import *
import copy
import pdb

def trial_precomputed(example_loss_table, example_size_table, lambdas_example_table, gamma, delta, num_lam, num_calib, batch_size, tlambda, bound_str):
    total=example_loss_table.shape[0]
    perm = torch.randperm(example_loss_table.shape[0])
    example_loss_table = example_loss_table[perm]
    example_size_table = example_size_table[perm]
    calib_losses, val_losses = (example_loss_table[0:num_calib], example_loss_table[num_calib:])
    calib_sizes, val_sizes = (example_size_table[0:num_calib], example_size_table[num_calib:])

    lhat = get_lhat_from_table(calib_losses[:,::-1], lambdas_example_table[::-1], gamma, delta, tlambda, bound_str)

    losses = val_losses[:,np.argmax(lambdas_example_table == lhat)]
    sizes = val_sizes[:,np.argmax(lambdas_example_table == lhat)]

    return losses.mean(), torch.tensor(sizes), lhat

def plot_histograms(df_list,gamma,delta,bounds_to_plot):
    fig, axs = plt.subplots(nrows=1,ncols=2,figsize=(12,3))

    minrisk = min([df['risk'].min() for df in df_list])
    maxrisk = max([df['risk'].max() for df in df_list])

    risk_bins = None#np.arange(minrisk, maxrisk, 0.0005) 
    ub = 20. # risk upper bound.
    
    for i in range(len(df_list)):
        df = df_list[i]
        axs[0].hist(np.array((df['risk'] * ub).tolist()), risk_bins, alpha=0.7, density=True)

        # Sizes will be 10 times as big as risk, since we pool it over runs.
        sizes = torch.cat(df['sizes'].tolist(),dim=0).numpy()
        #d = np.diff(np.unique(sizes)).min()
        #lofb = sizes.min() - float(d)/2
        #rolb = sizes.max() + float(d)/2
        #bins = np.arange(lofb,rolb+d, d)
        axs[1].hist(sizes, label='RCPS-' + bounds_to_plot[i], alpha=0.7, density=True)
    
    axs[0].set_xlabel(r'risk ($\AA$)')
    axs[0].locator_params(axis='x', nbins=8)
    axs[0].set_ylabel('density')
    #axs[0].set_yticks([0,100])
    axs[0].axvline(x=gamma*ub,c='#999999',linestyle='--',alpha=0.7)
    axs[1].set_xlabel('size')
    axs[1].set_yscale('log')
    sns.despine(ax=axs[0],top=True,right=True)
    sns.despine(ax=axs[1],top=True,right=True)
    #axs[1].set_xlim([-0.5,rolb])
    axs[1].legend()
    plt.tight_layout()
    plt.savefig( (f'outputs/histograms/{gamma}_{delta}_{num_calib}_protein_histograms').replace('.','_') + '.pdf')

def get_example_loss_and_size_tables(ls_probs, ls_preds, ls_gt, ls_targets, ls_dists, lambdas_example_table, num_calib):
    lam_len = len(lambdas_example_table)
    lam_low = min(lambdas_example_table)
    lam_high = max(lambdas_example_table)
    fname_loss = f'./.cache/{lam_low}_{lam_high}_{lam_len}_example_loss_table.npy'
    fname_size = f'./.cache/{lam_low}_{lam_high}_{lam_len}_example_size_table.npy'
    try:
        loss_table = np.load(fname_loss)
        size_table = np.load(fname_size)
    except:
        loss_table = np.zeros((len(ls_probs), lam_len))
        size_table = np.zeros((len(ls_probs), lam_len))
        print('Calculating loss and size table. (One time only.)')
        for j in tqdm(range(lam_len)):
            ls_sets = ls_sets_from_probs(ls_probs, ls_preds, ls_dists, lambdas_example_table[j]) 
            loss_table[:,j] = arr_l2_loss(ls_sets, ls_gt) 
            size_table[:,j] = np.array([np.random.choice((s[1]-s[0]).flatten()) for s in ls_sets]) # TODO:I randomly sample here, but that's only one possible choice. 

        np.save(fname_loss, loss_table)
        np.save(fname_size, size_table)

    return loss_table, size_table

def experiment(gamma,delta,lambdas_example_table,num_lam,num_calib,num_grid_hbb,ub,ub_sigma,epsilon,num_trials,maxiters,bounds_to_plot,batch_size=128,casp13_alphafoldv1_data_dir):
    df_list = []
    for bound_str in bounds_to_plot:
        if bound_str == 'Bentkus':
            bound_fn = bentkus_mu_plus
        elif bound_str == 'CLT':
            bound_fn = None 
        elif bound_str == 'HB':
            bound_fn = HB_mu_plus
        elif bound_str == 'HBB':
            bound_fn = HBB_mu_plus
        elif bound_str == 'WSR':
            bound_fn = WSR_mu_plus
        else:
            raise NotImplemented
        fname = f'.cache/{gamma}_{delta}_{num_lam}_{num_calib}_{num_trials}_{bound_str}_hierarchical_dataframe.pkl'

        df = pd.DataFrame(columns = ["$\\hat{\\lambda}$","risk","sizes","gamma","delta"])
        try:
            df = pd.read_pickle(fname)
        except FileNotFoundError:
            
            ls_probs, ls_preds, ls_gt, ls_targets, ls_dists = get_preds_gt(casp13_alphafoldv1_data_dir)

            with torch.no_grad():
                # get the precomputed binary search
                tlambda = get_tlambda(num_lam,deltas,num_calib,num_grid_hbb,ub,ub_sigma,epsilon,maxiters,bound_str,bound_fn)
                example_loss_table, example_size_table = get_example_loss_and_size_tables(ls_probs, ls_preds, ls_gt, ls_targets, ls_dists, lambdas_example_table, num_calib)
                example_loss_table = example_loss_table / 20. # it's bounded above by 20.

                for i in tqdm(range(num_trials)):
                    risk, sizes, lhat = trial_precomputed(example_loss_table, example_size_table, lambdas_example_table, gamma, delta, num_lam, num_calib, batch_size, tlambda, bound_str)
                    df = df.append({"$\\hat{\\lambda}$": lhat,
                                    "risk": risk,
                                    "sizes": sizes,
                                    "gamma": gamma,
                                    "delta": delta}, ignore_index=True)
            df.to_pickle(fname)
        df_list = df_list + [df]

    plot_histograms(df_list,gamma,delta,bounds_to_plot)

if __name__ == "__main__":
    sns.set(palette='pastel',font='serif')
    sns.set_style('white')
    fix_randomness(seed=0)

    bounds_to_plot = ['CLT']
    casp13_alphafoldv1_data_dir = '/scratch/aa/casp13-data/'
        
    gammas = [0.1]
    deltas = [0.1]
    params = list(zip(gammas,deltas))
    num_lam = 1500 
    num_calib = 35 
    num_grid_hbb = 200
    epsilon = 1e-10 
    maxiters = int(1e5)
    num_trials = 1000 
    ub = 0.2
    ub_sigma = np.sqrt(2)
    lambdas_example_table = np.linspace(0,1,1000)
    
    deltas_precomputed = [0.001, 0.01, 0.05, 0.1]
    
    for gamma, delta in params:
        print(f"\n\n\n ============           NEW EXPERIMENT gamma={gamma} delta={delta}           ============ \n\n\n") 
        experiment(gamma,delta,lambdas_example_table,num_lam,num_calib,num_grid_hbb,ub,ub_sigma,epsilon,num_trials,maxiters,bounds_to_plot,casp13_alphafoldv1_data_dir)
