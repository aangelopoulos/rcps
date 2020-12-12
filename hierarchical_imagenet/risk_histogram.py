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
from conformal import platt_logits, ConformalModelScores
import pdb

def get_lamhat_precomputed(scores, labels, losses, gamma, delta, num_lam, num_calib, tlambda):
    lams = torch.linspace(0,1,num_lam)
    lam = None
    labels_onehot = torch.nn.functional.one_hot(labels,scores.shape[1])
    for i in range(lams.shape[0]):
        lam = lams[i]
        est_labels_onehot = (losses.view(1,-1) * scores > lam).to(float) 
        true_loss = (losses.view(1,-1) * torch.nn.functional.relu(labels_onehot - est_labels_onehot)).sum(dim=1)
        Rhat = true_loss.mean()
        sigmahat = true_loss.std()
        if Rhat >= gamma:
            break
        if Rhat + tlambda(Rhat,sigmahat,delta) >= gamma:
            break

    return lam 

def conformal_trial_precomputed(scores,labels,losses,gamma,delta,num_calib,batch_size,randomized=True):
    total=scores.shape[0]
    perm = torch.randperm(scores.shape[0])
    scores = scores[perm]
    labels = labels[perm]
    calib_scores, val_scores = (scores[0:num_calib], scores[num_calib:])
    calib_labels, val_labels = (labels[0:num_calib], labels[num_calib:])
    # make datasets
    dset_cal = torch.utils.data.TensorDataset(calib_scores,calib_labels.long())
    dset_val = torch.utils.data.TensorDataset(val_scores,val_labels.long())

    # Prepare the loaders
    loader_cal = torch.utils.data.DataLoader(dset_cal, batch_size = batch_size, shuffle=False, pin_memory=True)
    loader_val = torch.utils.data.DataLoader(dset_val, batch_size = batch_size, shuffle=False, pin_memory=True)
    # Conformalize the model
    conformal_model = ConformalModelScores(None, loader_cal, alpha=gamma, randomized=randomized, naive=False) # RAPS
    # Collect results
    risks_avg, sizes = validate(loader_val, conformal_model, losses, print_bool=False)
    sizes = torch.tensor(np.concatenate(sizes,axis=0))
    return risks_avg, sizes, 0 

def trial_precomputed(scores, labels, losses, gamma,delta,num_lam,num_calib,batch_size,tlambda):
    total=scores.shape[0]
    perm = torch.randperm(scores.shape[0])
    scores = scores[perm]
    labels = labels[perm]
    calib_scores, val_scores = (scores[0:num_calib], scores[num_calib:])
    calib_labels, val_labels = (labels[0:num_calib], labels[num_calib:])

    lhat = get_lamhat_precomputed(calib_scores, calib_labels, losses, gamma, delta, num_lam, num_calib, tlambda)
    est_labels = (val_scores > lhat).to(float)
    risks, sizes = get_metrics_precomputed(est_labels,val_labels,losses,scores.shape[1])
    return risks.mean(), sizes, lhat.item()

def plot_histograms(df_list,gamma,delta,bounds_to_plot):
    fig, axs = plt.subplots(nrows=1,ncols=2,figsize=(12,3))

    minrisk = min([df['risk'].min() for df in df_list])
    maxrisk = min([df['risk'].max() for df in df_list])

    risk_bins = np.arange(minrisk, maxrisk, 0.001) 
    
    for i in range(len(df_list)):
        df = df_list[i]
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
        axs[1].hist(sizes, np.arange(lofb,rolb+d, d), label=bounds_to_plot[i], alpha=0.7, density=True)
    
    axs[0].set_xlabel('risk')
    axs[0].locator_params(axis='x', nbins=5)
    axs[0].set_ylabel('density')
    axs[0].set_yticks([0,100])
    axs[0].axvline(x=gamma,c='#999999',linestyle='--',alpha=0.7)
    axs[1].set_xlabel('size')
    sns.despine(ax=axs[0],top=True,right=True)
    sns.despine(ax=axs[1],top=True,right=True)
    if 'Conformal' not in bounds_to_plot:
        axs[1].set_xlim([0.5,rolb])
    if len(bounds_to_plot) > 1:
        axs[1].legend()
    plt.tight_layout()
    plt.savefig( (f'outputs/histograms/{gamma}_{delta}_{num_calib}_imagenet_histograms').replace('.','_') + '.pdf')

def experiment(losses,gamma,delta,num_lam,num_calib,num_grid_hbb,ub,ub_sigma,epsilon,num_trials,maxiters,bounds_to_plot, batch_size=128):
    df_list = []
    for bound_str in bounds_to_plot:
        if bound_str == 'Bentkus':
            bound_fn = bentkus_mu_plus
        elif bound_str == 'HBB':
            bound_fn = HBB_mu_plus
        elif bound_str == 'Conformal':
            bound_fn = None
        else:
            raise NotImplemented
        fname = f'.cache/{gamma}_{delta}_{num_lam}_{num_calib}_{num_trials}_{bound_str}_dataframe.pkl'

        df = pd.DataFrame(columns = ["$\\hat{\\lambda}$","risk","sizes","gamma","delta"])
        try:
            df = pd.read_pickle(fname)
        except FileNotFoundError:
            dataset_precomputed = get_logits_dataset('ResNet152', 'Imagenet', '/scratch/group/ilsvrc/val/')
            print('Dataset loaded')
            
            classes_array = get_imagenet_classes()
            T = platt_logits(dataset_precomputed)
            
            logits, labels = dataset_precomputed.tensors
            scores = (logits/T.cpu()).softmax(dim=1)

            with torch.no_grad():
                # get the precomputed binary search
                if bound_str != 'Conformal':
                    tlambda = get_tlambda(num_lam,deltas,num_calib,num_grid_hbb,ub,ub_sigma,epsilon,maxiters,bound_str,bound_fn)

                for i in tqdm(range(num_trials)):
                    if bound_str == 'Conformal':
                        risk, sizes, lhat = conformal_trial_precomputed(scores,labels,losses,gamma,delta,num_calib,batch_size)
                    else:
                        risk, sizes, lhat = trial_precomputed(scores,labels,losses,gamma,delta,num_lam,num_calib,batch_size,tlambda)
                    df = df.append({"$\\hat{\\lambda}$": lhat,
                                    "risk": risk,
                                    "sizes": sizes,
                                    "gamma": gamma,
                                    "delta": delta}, ignore_index=True)
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

    bounds_to_plot = ['HBB']

    losses = torch.ones((1000,))
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
    
    deltas_precomputed = [0.001, 0.01, 0.05, 0.1]
    
    for gamma, delta in params:
        print(f"\n\n\n ============           NEW EXPERIMENT gamma={gamma} delta={delta}           ============ \n\n\n") 
        experiment(losses,gamma,delta,num_lam,num_calib,num_grid_hbb,ub,ub_sigma,epsilon,num_trials,maxiters,bounds_to_plot)
