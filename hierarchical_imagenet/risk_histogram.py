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
from scipy.optimize import brentq
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

def get_heights(st, scores, labels, idx_dict, name_dict):
    heights = np.zeros((len(st),)) 
    starting_nodes = scores.argmax(dim=1)
    for i in range(len(st)):
        st_leaf = idx_dict[starting_nodes[i].item()]
        heights[i] = len(st_leaf.parents) - len(st[i].parents) 
    return heights

def get_subtree(scores, lam, idx_dict, name_dict, memo):
    start = torch.argmax(scores, dim=1).numpy() 
    st = [copy.deepcopy(idx_dict[s]) for s in start] # subtrees

    for i in range(start.shape[0]):
        parent_index = 0
        curr_sos = subtree_sum_scores(memo, i, name_dict[st[i].parents[parent_index]], scores[i], name_dict)
        if (i % 100) == 0:
            print(f'{i}\r', end='')
        while parent_index < len(st[i].parents) and curr_sos > lam:  # every iterate through the loop, the set shrinks.
            parent_index += 1
            curr_sos = subtree_sum_scores(memo, i, name_dict[st[i].parents[min(parent_index,len(st[i].parents)-1)]], scores[i], name_dict) 
        st[i] = name_dict[st[i].parents[min(parent_index,len(st[i].parents)-1)]]
    return st
    
def trial_precomputed(example_loss_table, example_height_table, lambdas_example_table, gamma, delta, num_lam, num_calib, batch_size, tlambda, bound_str):
    total=example_loss_table.shape[0]
    perm = torch.randperm(example_loss_table.shape[0])
    example_loss_table = example_loss_table[perm]
    example_height_table = example_height_table[perm]
    calib_losses, val_losses = (example_loss_table[0:num_calib], example_loss_table[num_calib:])
    calib_heights, val_heights = (example_height_table[0:num_calib], example_height_table[num_calib:])

    lhat = get_lhat_from_table(calib_losses, lambdas_example_table, gamma, delta, tlambda, bound_str)

    losses = val_losses[:,np.argmax(lambdas_example_table == lhat)]
    heights = val_heights[:,np.argmax(lambdas_example_table == lhat)]

    return losses.mean(), torch.tensor(heights), lhat

def plot_histograms(df_list,gamma,delta,bounds_to_plot):
    fig, axs = plt.subplots(nrows=1,ncols=2,figsize=(12,3))

    minrisk = min([df['risk'].min() for df in df_list])
    maxrisk = min([df['risk'].max() for df in df_list])

    risk_bins = np.arange(minrisk, maxrisk, 0.0005) 
    
    for i in range(len(df_list)):
        df = df_list[i]
        axs[0].hist(np.array(df['risk'].tolist()), risk_bins, alpha=0.7, density=True)

        # Sizes will be 10 times as big as risk, since we pool it over runs.
        sizes = torch.cat(df['heights'].tolist(),dim=0).numpy()
        d = np.diff(np.unique(sizes)).min()
        lofb = sizes.min() - float(d)/2
        rolb = sizes.max() + float(d)/2
        axs[1].hist(sizes, np.arange(lofb,rolb+d, d), label='RCPS-' + bounds_to_plot[i], alpha=0.7, density=True)
    
    axs[0].set_xlabel('risk')
    axs[0].locator_params(axis='x', nbins=8)
    axs[0].set_ylabel('density')
    axs[0].set_yticks([0,100])
    axs[0].axvline(x=gamma,c='#999999',linestyle='--',alpha=0.7)
    axs[1].set_xlabel('height')
    axs[1].set_yscale('log')
    sns.despine(ax=axs[0],top=True,right=True)
    sns.despine(ax=axs[1],top=True,right=True)
    axs[1].set_xlim([-0.5,rolb])
    axs[1].legend()
    plt.tight_layout()
    plt.savefig( (f'outputs/histograms/{gamma}_{delta}_{num_calib}_protein_histograms').replace('.','_') + '.pdf')

def load_imagenet_tree():
    with open('./wordnet_hierarchy.json', 'r') as file:
        data = file.read()
    imagenet_dict = json.loads(data)
    t = dict2tree(imagenet_dict)
    idx_dict = getIndexDict(t)
    name_dict = getNameDict(t)
    return idx_dict, name_dict

def get_example_loss_and_height_tables(scores, labels, idx_dict, name_dict, lambdas_example_table, num_calib):
    lam_len = len(lambdas_example_table)
    lam_low = min(lambdas_example_table)
    lam_high = max(lambdas_example_table)
    fname_loss = f'./.cache/{lam_low}_{lam_high}_{lam_len}_example_loss_table.npy'
    fname_height = f'./.cache/{lam_low}_{lam_high}_{lam_len}_example_height_table.npy'
    try:
        loss_table = np.load(fname_loss)
        height_table = np.load(fname_height)
    except:
        loss_table = np.zeros((scores.shape[0], lam_len))
        height_table = np.zeros((scores.shape[0], lam_len))
        memo = {}
        for j in range(lam_len):
            sts = get_subtree(scores, lambdas_example_table[j], idx_dict, name_dict, memo)
            losses_lam = hierarchical_loss(sts,labels,idx_dict,name_dict)
            loss_table[:,j] = losses_lam
            height_table[:,j] = get_heights(sts, scores, labels, idx_dict, name_dict)

        np.save(fname_loss, loss_table)
        np.save(fname_height, height_table)

    return loss_table, height_table

def experiment(losses,gamma,delta,lambdas_example_table,num_lam,num_calib,num_grid_hbb,ub,ub_sigma,epsilon,num_trials,maxiters,bounds_to_plot,batch_size=128,imagenet_val_dir):
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

        df = pd.DataFrame(columns = ["$\\hat{\\lambda}$","risk","heights","gamma","delta"])
        try:
            df = pd.read_pickle(fname)
        except FileNotFoundError:
            dataset_precomputed = get_logits_dataset('ResNet152', 'Imagenet', imagenet_val_dir)
            print('Dataset loaded')
            
            classes_array = get_imagenet_classes()
            T = platt_logits(dataset_precomputed)
            
            logits, labels = dataset_precomputed.tensors
            scores = (logits/T.cpu()).softmax(dim=1)

            idx_dict, name_dict = load_imagenet_tree()

            with torch.no_grad():
                # get the precomputed binary search
                tlambda = get_tlambda(num_lam,deltas,num_calib,num_grid_hbb,ub,ub_sigma,epsilon,maxiters,bound_str,bound_fn)
                example_loss_table, example_height_table = get_example_loss_and_height_tables(scores, labels, idx_dict, name_dict, lambdas_example_table, num_calib)

                for i in tqdm(range(num_trials)):
                    risk, heights, lhat = trial_precomputed(example_loss_table, example_height_table, lambdas_example_table, gamma, delta, num_lam, num_calib, batch_size, tlambda, bound_str)
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
    imagenet_val_dir = '/scratch/group/ilsvrc/val/'

    bounds_to_plot = ['CLT', 'HB', 'WSR']

    losses = torch.ones((1000,))
    gammas = [0.05]
    deltas = [0.1]
    params = list(zip(gammas,deltas))
    num_lam = 1500 
    num_calib = 30000 
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
        experiment(losses,gamma,delta,lambdas_example_table,num_lam,num_calib,num_grid_hbb,ub,ub_sigma,epsilon,num_trials,maxiters,bounds_to_plot,imagenet_val_dir)
