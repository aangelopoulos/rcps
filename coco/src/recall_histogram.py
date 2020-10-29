import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
import torch
import torchvision as tv
from asl.helper_functions.helper_functions import parse_args
from asl.loss_functions.losses import AsymmetricLoss, AsymmetricLossOptimized
from asl.models import create_model
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
from concentration import *
import pdb

parser = argparse.ArgumentParser(description='ASL MS-COCO predictor')

parser.add_argument('--model_path',type=str,default='../models/MS_COCO_TResNet_xl_640_88.4.pth')
parser.add_argument('--dset_path',type=str,default='../data/')
parser.add_argument('--model_name',type=str,default='tresnet_xl')
parser.add_argument('--input_size',type=int,default=640)
parser.add_argument('--dataset_type',type=str,default='MS-COCO')
parser.add_argument('--batch_size',type=int,default=5000)
parser.add_argument('--th',type=float,default=0.7)

def get_lamhat_precomputed(scores, labels, gamma, delta, num_lam, num_calib, tlambda):
    lams = torch.linspace(0,1,num_lam)
    lam = None
    for i in range(lams.shape[0]):
        lam = lams[i]
        est_labels = (scores > lam).to(float) 
        avg_acc = (est_labels * labels.to(float)/labels.sum()).sum()
        Rhat = 1-avg_acc
        sigmahat = (1-(est_labels * labels.to(float)/labels.sum(dim=1).unsqueeze(1)).mean(dim=1)).std()
        if Rhat >= gamma:
            break
        if Rhat + tlambda(Rhat,sigmahat,delta) >= gamma:
            break

    return lam 

def trial_precomputed(scores,labels,gamma,delta,num_lam,num_calib,batch_size,tlambda):
    total=scores.shape[0]
    perm = torch.randperm(scores.shape[0])
    scores = scores[perm]
    labels = labels[perm]
    calib_scores, val_scores = (scores[0:num_calib], scores[num_calib:])
    calib_labels, val_labels = (labels[0:num_calib], labels[num_calib:])

    lhat = get_lamhat_precomputed(calib_scores, calib_labels, gamma, delta, num_lam, num_calib, tlambda)
    est_labels = (val_scores > lhat).to(float)
    prec, rec, size = get_metrics_precomputed(est_labels,val_labels)
    return prec.mean().item(), rec.mean().item(), size, lhat.item()

def plot_histograms(df_list,gamma,delta,bounds_to_plot):
    rig, axs = plt.subplots(nrows=1,ncols=2,figsize=(12,3))

    minrecall = min([df['recall'].min() for df in df_list])
    maxrecall = min([df['recall'].max() for df in df_list])

    recall_bins = np.arange(minrecall, maxrecall, 0.002) 
    
    for i in range(len(df_list)):
        df = df_list[i]
        axs[0].hist(np.array(df['recall'].tolist()), recall_bins, alpha=0.7, density=True)

        # Sizes will be 10 times as big as recall, since we pool it over runs.
        sizes = torch.cat(df['size'].tolist(),dim=0).numpy()
        d = np.diff(np.unique(sizes)).min()
        lofb = sizes.min() - float(d)/2
        rolb = sizes.max() + float(d)/2
        axs[1].hist(sizes, np.arange(lofb,rolb+d, d), label=bounds_to_plot[i], alpha=0.7, density=True)
    
    axs[0].set_xlabel('risk')
    axs[0].locator_params(axis='x', nbins=4)
    axs[0].set_ylabel('density')
    axs[0].set_yticks([0,100])
    axs[1].set_xlabel('size')
    sns.despine(ax=axs[0],top=True,right=True)
    sns.despine(ax=axs[1],top=True,right=True)
    #axs[1].legend()
    plt.tight_layout()
    plt.savefig('../' + (f'outputs/histograms/{gamma}_{delta}_histograms').replace('.','_') + '.pdf')

def experiment(gamma,delta,num_lam,num_calib,num_grid_hbb,ub,ub_sigma,epsilon,num_trials,maxiters,bounds_to_plot):
    df_list = []
    for bound_str in bounds_to_plot:
        if bound_str == 'Bentkus':
            bound_fn = bentkus_mu_plus
        elif bound_str == 'HBB':
            bound_fn = HBB_mu_plus
        else:
            raise NotImplemented
        fname = f'../.cache/{gamma}_{delta}_{bound_str}_dataframe.pkl'

        df = pd.DataFrame(columns = ["$\\hat{\\lambda}$","precision","recall","size","gamma","delta"])
        try:
            df = pd.read_pickle(fname)
        except FileNotFoundError:
            dataset = tv.datasets.CocoDetection('../data/val2017','../data/annotations_trainval2017/instances_val2017.json',transform=tv.transforms.Compose([tv.transforms.Resize((args.input_size, args.input_size)),
                                                                                                                                                             tv.transforms.ToTensor()]))
            print('Dataset loaded')
            
            #model
            state = torch.load('../models/MS_COCO_TResNet_xl_640_88.4.pth', map_location='cpu')
            classes_list = np.array(list(state['idx_to_class'].values()))
            args.num_classes = state['num_classes']
            model = create_model(args).cuda()
            model.load_state_dict(state['model'], strict=True)
            model.eval()
            print('Model Loaded')
            corr = get_correspondence(classes_list,dataset.coco.cats)

            # get dataset
            dataset_fname = '../.cache/coco_val.pkl'
            if os.path.exists(dataset_fname):
                dataset_precomputed = pkl.load(open(dataset_fname,'rb'))
                print(f"Precomputed dataset loaded. Size: {len(dataset_precomputed)}")
            else:
                dataset_precomputed = get_scores_targets(model, torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=True), corr)
                pkl.dump(dataset_precomputed,open(dataset_fname,'wb'),protocol=pkl.HIGHEST_PROTOCOL)
            scores, labels = dataset_precomputed.tensors

            # get the precomputed binary search
            tlambda = get_tlambda(num_lam,deltas,num_calib,num_grid_hbb,ub,ub_sigma,epsilon,maxiters,bound_str,bound_fn)

            for i in tqdm(range(num_trials)):
                prec, rec, sz, lhat = trial_precomputed(scores,labels,gamma,delta,num_lam,num_calib,args.batch_size,tlambda)
                df = df.append({"$\\hat{\\lambda}$": lhat,
                                "precision": prec,
                                "recall": rec,
                                "size": sz,
                                "gamma": gamma,
                                "delta": delta}, ignore_index=True)
            df.to_pickle(fname)
        df_list = df_list + [df]

    plot_histograms(df_list,gamma,delta,bounds_to_plot)


if __name__ == "__main__":
    with torch.no_grad():
        sns.set(palette='pastel',font='serif')
        sns.set_style('white')
        fix_randomness(seed=0)
        args = parse_args(parser)

        bounds_to_plot = ['HBB']

        gammas = [0.1,0.05]
        deltas = [0.1,0.1]
        params = list(zip(gammas,deltas))
        num_lam = 1500 
        num_calib = 4000 
        num_grid_hbb = 200
        epsilon = 1e-10 
        maxiters = int(1e5)
        num_trials = 1000
        ub = 0.2
        ub_sigma = np.sqrt(2)
        
        for gamma, delta in params:
            print(f"\n\n\n ============           NEW EXPERIMENT gamma={gamma} delta={delta}           ============ \n\n\n") 
            experiment(gamma,delta,num_lam,num_calib,num_grid_hbb,ub,ub_sigma,epsilon,num_trials,maxiters,bounds_to_plot)
