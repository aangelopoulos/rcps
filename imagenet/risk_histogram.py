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
from utils import *
import seaborn as sns
from core.concentration import *
import pdb

def get_example_loss_and_size_tables(scores, labels, losses, lambdas_example_table, num_calib):
    lam_len = len(lambdas_example_table)
    lam_low = min(lambdas_example_table)
    lam_high = max(lambdas_example_table)
    fname_loss = f'.cache/{lam_low}_{lam_high}_{lam_len}_example_loss_table.npy'
    fname_sizes = f'.cache/{lam_low}_{lam_high}_{lam_len}_example_size_table.npy'
    try:
        loss_table = np.load(fname_loss)
        sizes_table = np.load(fname_sizes)
    except:
        loss_table = np.zeros((scores.shape[0], lam_len))
        sizes_table = np.zeros((scores.shape[0], lam_len))
        for j in range(lam_len):
            est_labels = (scores > lambdas_example_table[j]).float()
            loss_table[:,j] = weighted_loss(est_labels, torch.nn.functional.one_hot(labels, scores.shape[1]), losses)
            sizes_table[:,j] = est_labels.sum(dim=1)

        np.save(fname_loss, loss_table)
        np.save(fname_sizes, sizes_table)

    return loss_table, sizes_table

def weighted_loss(est_labels, labels, losses):
    current_labels = np.argmax(labels, axis=1)
    mask = (est_labels[np.arange(len(labels)), current_labels] == 0)
    weighted_loss = np.where(mask, losses[current_labels], 0)
    return weighted_loss

def trial_precomputed(example_loss_table, example_size_table, lambdas_example_table, gamma, delta, num_lam, num_calib, batch_size, tlambda, bound_str):
    total=example_loss_table.shape[0]
    perm = torch.randperm(example_loss_table.shape[0])
    example_loss_table = example_loss_table[perm]
    example_size_table = example_size_table[perm]
    calib_losses, val_losses = (example_loss_table[0:num_calib], example_loss_table[num_calib:])
    calib_sizes, val_sizes = (example_size_table[0:num_calib], example_size_table[num_calib:])

    lhat_rcps = get_lhat_from_table(calib_losses, lambdas_example_table, gamma, delta, tlambda, bound_str)

    losses_rcps = val_losses[:,np.argmax(lambdas_example_table == lhat_rcps)]
    sizes_rcps = val_sizes[:,np.argmax(lambdas_example_table == lhat_rcps)]

    return losses_rcps.mean(), torch.tensor(sizes_rcps), lhat_rcps

def plot_histograms(df_list,gamma,delta,bounds_to_plot):
    fig, axs = plt.subplots(nrows=1,ncols=2,figsize=(12,3))

    minrisk = min([df['risk'].min() for df in df_list])
    maxrisk = min([df['risk'].max() for df in df_list])

    risk_bins = np.arange(minrisk, maxrisk, 0.001)

    for i in range(len(df_list)):
        df = df_list[i]
        print(f"gamma:{gamma}, delta:{delta}, bound:{bounds_to_plot[i]}, coverage:{(df.risk > gamma).mean()}")
        # Use the same binning for everybody except conformal
        if bounds_to_plot[i] == 'Conformal':
            axs[0].hist(np.array(df['risk'].tolist()), None, alpha=0.7, density=True)
        else:
            axs[0].hist(np.array(df['risk'].tolist()), risk_bins, alpha=0.7, density=True)

        # Sizes will be 10 times as big as risk, since we pool it over runs.
        sizes = torch.cat(df['sizes'].tolist(),dim=0).numpy()
        d = np.diff(np.unique(sizes)).min()
        lofb = sizes.min() - float(d)/2
        rolb = sizes.max() + float(d)/2
        axs[1].hist(sizes, np.arange(lofb,rolb+d, d), label='RCPS-' + bounds_to_plot[i], alpha=0.7, density=True)

    axs[0].set_xlabel('risk')
    axs[0].locator_params(axis='x', nbins=5)
    axs[0].set_ylabel('density')
    axs[0].set_yticks([0,100])
    axs[0].axvline(x=gamma,c='#999999',linestyle='--',alpha=0.7)
    axs[1].set_xlabel('size')
    axs[1].legend()
    sns.despine(ax=axs[0],top=True,right=True)
    sns.despine(ax=axs[1],top=True,right=True)
    if 'Conformal' not in bounds_to_plot:
        axs[1].set_xlim([0.5,rolb])
    if len(bounds_to_plot) > 1:
        axs[1].legend()
    plt.tight_layout()
    plt.savefig( (f'outputs/histograms/{gamma}_{delta}_{num_calib}_imagenet_histograms').replace('.','_') + '.pdf')

def experiment(losses,gamma,delta,num_lam,num_calib,num_grid_hbb,ub,ub_sigma,lambdas_example_table,epsilon,num_trials,maxiters,bounds_to_plot, batch_size=128, imagenet_val_dir):
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
        elif bound_str == 'Conformal':
            bound_fn = None
        else:
            raise NotImplemented
        fname = f'.cache/{gamma}_{delta}_{num_lam}_{num_calib}_{num_trials}_{bound_str}_dataframe.pkl'

        df = pd.DataFrame(columns = ["$\\hat{\\lambda}$","risk","sizes","gamma","delta"])
        try:
            df = pd.read_pickle(fname)
        except FileNotFoundError:
            dataset_precomputed = get_logits_dataset('ResNet152', 'Imagenet', imagenet_val_dir)
            print('Dataset loaded')

            classes_array = get_imagenet_classes()
            T = platt_logits(dataset_precomputed)

            logits, labels = dataset_precomputed.tensors
            scores = (logits/T.cpu()).softmax(dim=1)

            with torch.no_grad():
                example_loss_table, example_size_table = get_example_loss_and_size_tables(scores, labels, losses, lambdas_example_table, num_calib)
                tlambda = get_tlambda(num_lam,deltas,num_calib,num_grid_hbb,ub,ub_sigma,epsilon,maxiters,bound_str.lower(), bound_fn)

                local_df_list = []
                for i in tqdm(range(num_trials)):
                    risk, sizes, lhat = trial_precomputed(example_loss_table, example_size_table, lambdas_example_table, gamma, delta, num_lam, num_calib, batch_size, tlambda, bound_str)
                    dict_local = {"$\\hat{\\lambda}$": lhat,
                                    "risk": risk,
                                    "sizes": [sizes],
                                    "gamma": gamma,
                                    "delta": delta
                                 }
                    df_local = pd.DataFrame(dict_local)
                    local_df_list = local_df_list + [df_local]
                df = pd.concat(local_df_list, axis=0, ignore_index=True)
                df.to_pickle(fname)

        df_list = df_list + [df]

    plot_histograms(df_list,gamma,delta,bounds_to_plot)

def platt_logits(calib_dataset, max_iters=10, lr=0.01, epsilon=0.01):
    calib_loader = torch.utils.data.DataLoader(calib_dataset, batch_size=1024, shuffle=False, pin_memory=True)
    nll_criterion = nn.CrossEntropyLoss().cuda()

    T = nn.Parameter(torch.Tensor([1.3]).cuda())

    optimizer = optim.SGD([T], lr=lr)
    for iter in range(max_iters):
        T_old = T.item()
        for x, targets in calib_loader:
            optimizer.zero_grad()
            x = x.cuda()
            x.requires_grad = True
            out = x/T
            loss = nll_criterion(out, targets.long().cuda())
            loss.backward()
            optimizer.step()
        if abs(T_old - T.item()) < epsilon:
            break
    return T

if __name__ == "__main__":
    sns.set(palette='pastel',font='serif')
    sns.set_style('white')
    fix_randomness(seed=0)

    bounds_to_plot = ['CLT','HB','WSR']
    imagenet_val_dir = '/scratch/group/ilsvrc/val' #TODO: Replace this with YOUR location of imagenet val set.

    losses = torch.rand((1000,))
    gammas = [0.1,0.05]
    deltas = [0.1,0.1]
    params = list(zip(gammas,deltas))
    num_lam = 1500
    num_calib = 30000
    num_grid_hbb = 200
    epsilon = 1e-10
    maxiters = int(1e5)
    num_trials = 1000
    ub = 0.2
    ub_sigma = np.sqrt(2)
    lambdas_example_table = np.flip(np.linspace(0,1,1000), axis=0)

    deltas_precomputed = [0.001, 0.01, 0.05, 0.1]

    for gamma, delta in params:
        print(f"\n\n\n ============           NEW EXPERIMENT gamma={gamma} delta={delta}           ============ \n\n\n")
        experiment(losses,gamma,delta,num_lam,num_calib,num_grid_hbb,ub,ub_sigma,lambdas_example_table,epsilon,num_trials,maxiters,bounds_to_plot,imagenet_val_dir)
