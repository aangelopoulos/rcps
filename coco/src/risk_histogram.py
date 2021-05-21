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
from core.concentration import *
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

def get_example_loss_and_size_tables(scores, labels, lambdas_example_table, num_calib):
    lam_len = len(lambdas_example_table)
    lam_low = min(lambdas_example_table)
    lam_high = max(lambdas_example_table)
    fname_loss = f'../.cache/{lam_low}_{lam_high}_{lam_len}_example_loss_table.npy'
    fname_sizes = f'../.cache/{lam_low}_{lam_high}_{lam_len}_example_size_table.npy'
    try:
        loss_table = np.load(fname_loss)
        sizes_table = np.load(fname_sizes)
    except:
        loss_table = np.zeros((scores.shape[0], lam_len))
        sizes_table = np.zeros((scores.shape[0], lam_len))
        print("caching loss and size tables")
        for j in tqdm(range(lam_len)):
            est_labels = scores > lambdas_example_table[j]
            loss, sizes = get_metrics_precomputed(est_labels, labels)
            loss_table[:,j] = loss 
            sizes_table[:,j] = sizes

        np.save(fname_loss, loss_table)
        np.save(fname_sizes, sizes_table)

    return loss_table, sizes_table

def trial_precomputed(example_loss_table, example_size_table, lambdas_example_table, gamma, delta, num_lam, num_calib, batch_size, tlambda, bound_str):
    #total=example_loss_table.shape[0]
    #perm = torch.randperm(example_loss_table.shape[0])
    #example_loss_table = example_loss_table[perm]
    #example_size_table = example_size_table[perm]
    rng_state = np.random.get_state()
    np.random.shuffle(example_loss_table)
    np.random.set_state(rng_state)
    np.random.shuffle(example_size_table)

    calib_losses, val_losses = (example_loss_table[0:num_calib], example_loss_table[num_calib:])
    calib_sizes, val_sizes = (example_size_table[0:num_calib], example_size_table[num_calib:])

    lhat_rcps = get_lhat_from_table_binarysearch(calib_losses, lambdas_example_table, gamma, delta, tlambda, bound_str)

    losses_rcps = val_losses[:,np.argmax(lambdas_example_table == lhat_rcps)]
    sizes_rcps = val_sizes[:,np.argmax(lambdas_example_table == lhat_rcps)]

    temp_calib_losses = calib_losses.copy()
    temp_calib_losses[temp_calib_losses > 0] = 1 # for the conformal baseline, use a 0-1 multiclass loss.
    lhat_conformal = get_lhat_conformal_from_table(temp_calib_losses, lambdas_example_table, gamma)

    losses_conformal = val_losses[:,np.argmax(lambdas_example_table == lhat_conformal)]
    sizes_conformal = val_sizes[:,np.argmax(lambdas_example_table == lhat_conformal)]
    conformal_coverage = (losses_conformal > 0).astype(np.float64).mean()
    
    return losses_rcps.mean(), torch.tensor(sizes_rcps), lhat_rcps, losses_conformal.mean(), torch.tensor(sizes_conformal), lhat_conformal, conformal_coverage

def plot_histograms(df_list,gamma,delta,bounds_to_plot):
    fig, axs = plt.subplots(nrows=1,ncols=2,figsize=(12,3))

    minrecall = min([min(df['risk_rcps'].min(),df['risk_conformal'].min()) for df in df_list])
    maxrecall = max([max(df['risk_rcps'].max(),df['risk_conformal'].max()) for df in df_list])

    recall_bins = np.arange(minrecall, maxrecall, 0.005) 
    
    sizes = torch.cat(df_list[0]['sizes_rcps'].tolist(),dim=0).numpy()
    bl_sizes = torch.cat(df_list[0]['sizes_conformal'].tolist(),dim=0).numpy()
    conformal_coverages = df_list[0]['conformal_coverage'].to_numpy() 
    print(f"Conformal coverage for baseline: {conformal_coverages.mean()}")
    all_sizes = np.concatenate((sizes,bl_sizes),axis=0)
    d = np.diff(np.unique(all_sizes)).min()
    lofb = all_sizes.min() - float(d)/2
    rolb = all_sizes.max() + float(d)/2

    for i in range(len(df_list)):
        df = df_list[i]
        print(f"Bound {bounds_to_plot[i]} has coverage {1-(df['risk_rcps'] > gamma).mean()}")
        axs[0].hist(np.array(df['risk_rcps'].tolist()), recall_bins, alpha=0.7, density=True, label=bounds_to_plot[i])

        # Sizes will be 10 times as big as recall, since we pool it over runs.
        sizes = torch.cat(df['sizes_rcps'].tolist(),dim=0).numpy()
        axs[1].hist(sizes, np.arange(lofb,rolb+d, d), label='RCPS-' + bounds_to_plot[i], alpha=0.7, density=True)
    
    axs[0].hist(np.array(df_list[0]['risk_conformal'].tolist()), recall_bins, alpha=0.7, density=True, label='Conformal')
    axs[1].hist(bl_sizes, np.arange(lofb,rolb+d, d), label='Conformal', alpha=0.7, density=True)
    axs[0].set_xlabel('risk')
    axs[0].locator_params(axis='x', nbins=4)
    axs[0].set_ylabel('density')
    axs[0].set_yticks([0,100])
    axs[0].axvline(x=gamma,c='#999999',linestyle='--',alpha=0.7)
    axs[1].set_xlabel('size')
    sns.despine(ax=axs[0],top=True,right=True)
    sns.despine(ax=axs[1],top=True,right=True)
    axs[1].legend()
    plt.tight_layout()
    plt.savefig('../' + (f'outputs/histograms/{gamma}_{delta}_coco_histograms').replace('.','_') + '.pdf')


def experiment(gamma,delta,num_lam,num_calib,num_grid_hbb,ub,ub_sigma,lambdas_example_table,epsilon,num_trials,maxiters,batch_size,bounds_to_plot, coco_val_2017_directory, coco_instances_val_2017_json):
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
        fname = f'../.cache/{gamma}_{delta}_{num_calib}_{bound_str}_{num_trials}_dataframe.pkl'

        df = pd.DataFrame(columns = ["$\\hat{\\lambda}$","risk_rcps","size_rcps","risk_conformal","size_conformal","gamma","delta"])
        try:
            df = pd.read_pickle(fname)
        except FileNotFoundError:
            dataset = tv.datasets.CocoDetection(coco_val_2017_directory,transform=tv.transforms.Compose([tv.transforms.Resize((args.input_size, args.input_size)),
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
            example_loss_table, example_size_table = get_example_loss_and_size_tables(scores, labels, lambdas_example_table, num_calib)
            tlambda = get_tlambda(num_lam,deltas,num_calib,num_grid_hbb,ub,ub_sigma,epsilon,maxiters,bound_str,bound_fn)
            
            local_df_list = []
            for i in tqdm(range(num_trials)):
                risk_rcps, sizes_rcps, lhat_rcps, risk_conformal, sizes_conformal, lhat_conformal, conformal_coverage = trial_precomputed(example_loss_table, example_size_table, lambdas_example_table, gamma, delta, num_lam, num_calib, batch_size, tlambda, bound_str)
                dict_local = {"$\\hat{\\lambda}$": lhat_rcps,
                                "risk_rcps": risk_rcps,
                                "sizes_rcps": [sizes_rcps],
                                "$\\hat{\\lambda}_{c}$": lhat_conformal,
                                "risk_conformal": risk_conformal,
                                "sizes_conformal": [sizes_conformal],
                                "conformal_coverage": conformal_coverage,
                                "gamma": gamma,
                                "delta": delta
                             }
                df_local = pd.DataFrame(dict_local)
                local_df_list = local_df_list + [df_local]
            df = pd.concat(local_df_list, axis=0, ignore_index=True)
            df.to_pickle(fname)
        df_list = df_list + [df]

    plot_histograms(df_list,gamma,delta,bounds_to_plot)


if __name__ == "__main__":
    with torch.no_grad():
        sns.set(palette='pastel',font='serif')
        sns.set_style('white')
        fix_randomness(seed=0)
        args = parse_args(parser)
        coco_val_2017_directory = '../data/val2017'
        coco_instances_val_2017_json = '../data/annotations_trainval2017/instances_val2017.json'

        bounds_to_plot = ['CLT','HB','WSR']

        gammas = [0.1,0.05]
        deltas = [0.1,0.1]
        params = list(zip(gammas,deltas))
        num_lam = 1500 
        num_calib = 2000 
        num_grid_hbb = 200
        epsilon = 1e-10 
        maxiters = int(1e5)
        num_trials = 1000 # should be 1000
        ub = 0.2
        ub_sigma = np.sqrt(2)
        lambdas_example_table = np.flip(np.linspace(0,1,10000), axis=0)
        
        deltas_precomputed = [0.001, 0.01, 0.05, 0.1]
        
        for gamma, delta in params:
            print(f"\n\n\n ============           NEW EXPERIMENT gamma={gamma} delta={delta}           ============ \n\n\n") 
            experiment(gamma,delta,num_lam,num_calib,num_grid_hbb,ub,ub_sigma,lambdas_example_table,epsilon,num_trials,maxiters,args.batch_size,bounds_to_plot,coco_val_2017_directory,coco_instances_val_2017_json)
