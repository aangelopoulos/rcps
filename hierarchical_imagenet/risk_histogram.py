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
from ntree import *
import copy
import pdb

def subtree_sum_scores(memo, i, st, score, name_dict): # one at a time; score is a vector. i is for memoization
    if memo != None  and  st.name + str(i) in memo:
        return memo[st.name + str(i)]
    else:
        # else
        sum_scores = score[st.index] if st.index >= 0 else 0 
        for child in st.children:
            sum_scores += subtree_sum_scores(memo, i, child, score, name_dict)
        if memo != None:
            memo[st.name + str(i)] = sum_scores # add to cache
        return sum_scores

def hierarchical_loss(st, labels, idx_dict, name_dict):
    B = getMaxDepth(name_dict[idx_dict[0].parents[0]], idx_dict, name_dict)
    dists = np.zeros((len(st),))
    l_node = [copy.deepcopy(idx_dict[int(l)]) for l in labels.numpy()]
    for i in range(len(st)):
        dists[i] = getSubtreeLeafDistance(st[i],l_node[i])/B
    return dists

def get_subtree(scores, lam, idx_dict, name_dict, memo):
    start = torch.argmax(scores, dim=1).numpy() 
    st = [copy.deepcopy(idx_dict[s]) for s in start] # subtrees

    for i in range(start.shape[0]):
        parent_index = 0
        curr_sos = subtree_sum_scores(memo, i, name_dict[st[i].parents[parent_index]], scores[i], name_dict)
        if (i % 100) == 0:
            print(f'{i}\r', end='')
        while parent_index < len(st[i].parents) and curr_sos > -lam:
            parent_index += 1
            curr_sos = subtree_sum_scores(memo, i, name_dict[st[i].parents[min(parent_index,len(st[i].parents)-1)]], scores[i], name_dict) # TODO: This min shouldn't be here
        st[i] = name_dict[st[i].parents[min(parent_index,len(st[i].parents)-1)]]
    return st

def get_lamhat_precomputed(scores, labels, idx_dict, name_dict, gamma, delta, num_lam, num_calib, tlambda, memo):
    lams = torch.linspace(-0.25,-0.15,20) # totally heuristic
    lam = None
    for i in range(lams.shape[0]):
        lam = lams[i]
        st = get_subtree(scores, lam, idx_dict, name_dict, memo)
        true_loss = hierarchical_loss(st, labels, idx_dict, name_dict)
        Rhat = true_loss.mean()
        sigmahat = true_loss.std()
        print(f"lam:{lam}, Rplust:{Rhat + tlambda(Rhat, sigmahat, delta)}")
        if Rhat >= gamma:
            break
        if Rhat + tlambda(Rhat,sigmahat,delta) >= gamma:
            break

    return lam 

def trial_precomputed(scores, labels, idx_dict, name_dict, gamma,delta,num_lam,num_calib,batch_size,tlambda):
    total=scores.shape[0]
    perm = torch.randperm(scores.shape[0])
    scores = scores[perm]
    labels = labels[perm]
    calib_scores, val_scores = (scores[0:num_calib], scores[num_calib:])
    calib_labels, val_labels = (labels[0:num_calib], labels[num_calib:])

    memo = {} # dict for memoizing the subtree sums.
    lhat = get_lamhat_precomputed(calib_scores, calib_labels, idx_dict, name_dict, gamma, delta, num_lam, num_calib, tlambda, memo)

    memo = None # no more memo.
    st = get_subtree(val_scores, lhat, idx_dict, name_dict, memo)

    losses = hierarchical_loss(st, val_labels, idx_dict, name_dict)

    heights = torch.tensor(np.array([len(s.children) for s in st]))

    return losses.mean(), heights, lhat.item()

def plot_histograms(df_list,gamma,delta,bounds_to_plot):
    fig, axs = plt.subplots(nrows=1,ncols=2,figsize=(12,3))

    minrisk = min([df['risk'].min() for df in df_list])
    maxrisk = min([df['risk'].max() for df in df_list])

    risk_bins = np.arange(minrisk, maxrisk, 0.001) 
    
    for i in range(len(df_list)):
        df = df_list[i]
        axs[0].hist(np.array(df['risk'].tolist()), risk_bins, alpha=0.7, density=True)

        # Sizes will be 10 times as big as risk, since we pool it over runs.
        sizes = torch.cat(df['heights'].tolist(),dim=0).numpy()
        d = np.diff(np.unique(sizes)).min()
        lofb = sizes.min() - float(d)/2
        rolb = sizes.max() + float(d)/2
        axs[1].hist(sizes, np.arange(lofb,rolb+d, d), label=bounds_to_plot[i], alpha=0.7, density=True)
    
    axs[0].set_xlabel('risk')
    axs[0].locator_params(axis='x', nbins=5)
    axs[0].set_ylabel('density')
    axs[0].set_yticks([0,100])
    axs[0].axvline(x=gamma,c='#999999',linestyle='--',alpha=0.7)
    axs[1].set_xlabel('height')
    sns.despine(ax=axs[0],top=True,right=True)
    sns.despine(ax=axs[1],top=True,right=True)
    axs[1].set_xlim([-0.5,rolb])
    plt.tight_layout()
    plt.savefig( (f'outputs/histograms/{gamma}_{delta}_{num_calib}_hierarchical_imagenet_histograms').replace('.','_') + '.pdf')

def load_imagenet_tree():
    with open('./mobilenet.json', 'r') as file:
        data = file.read()
    imagenet_dict = json.loads(data)
    t = dict2tree(imagenet_dict)
    idx_dict = getIndexDict(t)
    name_dict = getNameDict(t)
    return idx_dict, name_dict

def experiment(losses,gamma,delta,num_lam,num_calib,num_grid_hbb,ub,ub_sigma,epsilon,num_trials,maxiters,bounds_to_plot, batch_size=128):
    df_list = []
    for bound_str in bounds_to_plot:
        if bound_str == 'Bentkus':
            bound_fn = bentkus_mu_plus
        elif bound_str == 'HBB':
            bound_fn = HBB_mu_plus
        else:
            raise NotImplemented
        fname = f'.cache/{gamma}_{delta}_{num_lam}_{num_calib}_{num_trials}_{bound_str}_hierarchical_dataframe.pkl'

        df = pd.DataFrame(columns = ["$\\hat{\\lambda}$","risk","heights","gamma","delta"])
        try:
            df = pd.read_pickle(fname)
        except FileNotFoundError:
            dataset_precomputed = get_logits_dataset('ResNet152', 'Imagenet', '/scratch/group/ilsvrc/val/')
            print('Dataset loaded')
            
            classes_array = get_imagenet_classes()
            T = platt_logits(dataset_precomputed)
            
            logits, labels = dataset_precomputed.tensors
            scores = (logits/T.cpu()).softmax(dim=1)

            idx_dict, name_dict = load_imagenet_tree()

            with torch.no_grad():
                # get the precomputed binary search
                tlambda = get_tlambda(num_lam,deltas,num_calib,num_grid_hbb,ub,ub_sigma,epsilon,maxiters,bound_str,bound_fn)

                for i in tqdm(range(num_trials)):
                    risk, heights, lhat = trial_precomputed(scores, labels, idx_dict, name_dict, gamma,delta,num_lam,num_calib,batch_size,tlambda)
                    df = df.append({"$\\hat{\\lambda}$": lhat,
                                    "risk": risk,
                                    "heights": heights,
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
    gammas = [0.05]
    deltas = [0.1]
    params = list(zip(gammas,deltas))
    num_lam = 1500 
    num_calib = 30000 
    num_grid_hbb = 200
    epsilon = 1e-10 
    maxiters = int(1e5)
    num_trials = 100 
    ub = 0.2
    ub_sigma = np.sqrt(2)
    
    deltas_precomputed = [0.001, 0.01, 0.05, 0.1]
    
    for gamma, delta in params:
        print(f"\n\n\n ============           NEW EXPERIMENT gamma={gamma} delta={delta}           ============ \n\n\n") 
        experiment(losses,gamma,delta,num_lam,num_calib,num_grid_hbb,ub,ub_sigma,epsilon,num_trials,maxiters,bounds_to_plot)
