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
from risk_histogram import get_lamhat_precomputed, get_subtree, hierarchical_loss, load_imagenet_tree, platt_logits 

def trial_precomputed(scores, labels, idx_dict, name_dict, gamma,delta,num_lam,num_calib,batch_size,tlambda):

    return losses.mean(), heights, lhat.item()

def experiment(losses,gamma,delta,num_lam,num_calib,num_grid_hbb,ub,ub_sigma,epsilon,maxiters,bound_str, batch_size=128):
    if bound_str == 'Bentkus':
        bound_fn = bentkus_mu_plus
    elif bound_str == 'HBB':
        bound_fn = HBB_mu_plus
    else:
        raise NotImplemented
    fname = f'.cache/{gamma}_{delta}_{num_lam}_{num_calib}_{bound_str}_hierarchical_grid.pkl'

    #dataset_precomputed = get_logits_dataset('ResNet152', 'Imagenet', '/scratch/group/ilsvrc/val/')
    #print('Dataset loaded')
    
    #classes_array = get_imagenet_classes()
    #T = platt_logits(dataset_precomputed)
    
    #logits, labels = dataset_precomputed.tensors
    #scores = (logits/T.cpu()).softmax(dim=1)
    with torch.no_grad():
        # get the precomputed binary search
        #tlambda = get_tlambda(num_lam,deltas,num_calib,num_grid_hbb,ub,ub_sigma,epsilon,maxiters,bound_str,bound_fn)

        # load imagenet hierarchy structure 
        idx_dict, name_dict = load_imagenet_tree()

        # Get the proper threshold
        #total=scores.shape[0]
        #perm = torch.randperm(scores.shape[0])
        #scores = scores[perm]
        #labels = labels[perm]
        #num_val = 1000
        #calib_scores, val_scores = (scores[0:num_calib], scores[num_calib:num_calib + num_val])
        #calib_labels, val_labels = (labels[0:num_calib], labels[num_calib:num_calib + num_val])

        # calibrate
        #memo = {} # dict for memoizing the subtree sums.
        lhat = -0.9#get_lamhat_precomputed(calib_scores, calib_labels, idx_dict, name_dict, gamma, delta, num_lam, num_calib, tlambda, memo)

        # validate
        #memo = None # no more memo.
        model = get_model('ResNet18')

        transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std =[0.229, 0.224, 0.225])
                        ])
        
        num_test = 500 
        num_plot = 8 
        dataset = torchvision.datasets.ImageFolder('/scratch/group/ilsvrc/val/', transform)

        memo = None 

        imgs = []
        val_scores = torch.zeros((num_test,1000))
        val_labels = torch.zeros((num_test,))
        rand_idxs = torch.randint(low=0, high=len(dataset), size=(num_test,))
        for i in range(num_test):
            ri = rand_idxs[i]
            img, lab = dataset[ri]
            imgs = imgs + [img]
            val_scores[i] = model(img[None,:]).softmax(dim=1)
            val_labels[i] = lab

        st = get_subtree(val_scores, lhat, idx_dict, name_dict, memo)

        losses = hierarchical_loss(st, val_labels, idx_dict, name_dict)

        heights = torch.tensor(np.array([len(s.children) for s in st]))

        top1 = torch.argmax(val_scores.softmax(dim=1),dim=1)
        top1_strings = [idx_dict[t.item()].name.split(',')[0] for t in top1]
        hier_strings = [s.name.split(',')[0] for s in st]
        true_strings = [idx_dict[v.item()].name.split(',')[0] for v in val_labels]
        correct = losses == 0
        top1_correct = top1 == val_labels

        imgs_to_plot = []
        top1_to_plot = []
        hier_to_plot = []
        true_strings_to_plot = []
        k = 0
        for i in range(len(top1_strings)):
            if correct[i] and not top1_correct[i] and k < num_plot:
                print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                img_to_plot = (imgs[i] * torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)) + torch.tensor([0.485, 0.456, 0.406]).view(-1,1,1)
                imgs_to_plot = imgs_to_plot + [img_to_plot.permute(1,2,0)]
                top1_to_plot = top1_to_plot + [[top1_strings[i], true_strings[i]]]
                hier_to_plot = hier_to_plot + [[hier_strings[i], true_strings[i]]]
                true_strings_to_plot = true_strings_to_plot + [[true_strings[i], top1_strings[i]]]
                k = k + 1

            print(f"true: {true_strings[i]}, hier: {hier_strings[i]}, top1: {top1_strings[i]}, correct: {correct[i]}, top1_correct: {top1_correct[i]}")

        gridplot_imgs(imgs_to_plot, hier_to_plot, true_strings_to_plot, 2, int(np.ceil(num_plot/2)))


def gridplot_imgs(imgs,est_labels,labels,rows,cols):
    fig, axs = plt.subplots(nrows=rows,ncols=cols,figsize=(cols*10,rows*10))
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    for idx, img in enumerate(imgs):
        r = idx//cols
        c = idx % cols
        axs[r,c].axis('off')
        axs[r,c].imshow(img, aspect='equal')
        corr_labelstr = ""
        est_labelstr = ""
        all_labelstr = ""
        fake_labelstr = ""
        num_labels = 0
        for i in range(len(est_labels[idx])):
            if est_labels[idx][i] in labels[idx]:
                corr_labelstr += est_labels[idx][i] + '\n'
                est_labelstr = '\n' + est_labelstr
                all_labelstr = '\n' + all_labelstr 
                fake_labelstr += est_labels[idx][i] + '\n'
            else:
                est_labelstr += est_labels[idx][i] + '\n'
                all_labelstr += '\n'
                fake_labelstr += est_labels[idx][i] + '\n'
            num_labels += 1


        for i in range(len(labels[idx])):
            if labels[idx][i] not in est_labels[idx]:
                all_labelstr += labels[idx][i] + '\n'
                fake_labelstr += labels[idx][i] + '\n'
                num_labels += 1

        # Remove last newline
        fake_labelstr = fake_labelstr[0:-1]
        all_labelstr = all_labelstr[0:-1]
        est_labelstr = est_labelstr[0:-1]
        corr_labelstr = corr_labelstr[0:-1] 

        # Resize text
        fontsize = 32
        if(num_labels <= 5):
            fontsize = 48

        # Make a fake bbox first.
        axs[r,c].text(0.05,0.95,fake_labelstr,transform=axs[r,c].transAxes,fontsize=fontsize,color='#00000000',verticalalignment='top',bbox=props)
        axs[r,c].text(0.05,0.95,all_labelstr,transform=axs[r,c].transAxes,fontsize=fontsize,color='#ff4555',verticalalignment='top')
        axs[r,c].text(0.05,0.95,est_labelstr,transform=axs[r,c].transAxes,fontsize=fontsize,color='#40B5BC',verticalalignment='top')
        axs[r,c].text(0.05,0.95,corr_labelstr,transform=axs[r,c].transAxes,fontsize=fontsize,color='k',verticalalignment='top')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05,hspace=0.05)
    plt.savefig('./outputs/grid_fig.pdf')

if __name__ == "__main__":
    sns.set(palette='pastel',font='serif')
    sns.set_style('white')
    fix_randomness(seed=0)

    bound_str = 'HBB'

    losses = torch.ones((1000,))
    gammas = [0.0]
    deltas = [0.01]
    params = list(zip(gammas,deltas))
    num_lam = 100 
    num_calib = 1000 
    num_grid_hbb = 200
    epsilon = 1e-10 
    maxiters = int(1e5)
    ub = 0.2
    ub_sigma = np.sqrt(2)
    
    deltas_precomputed = [0.001, 0.01, 0.05, 0.1]
    
    for gamma, delta in params:
        print(f"\n\n\n ============           NEW EXPERIMENT gamma={gamma} delta={delta}           ============ \n\n\n") 
        experiment(losses,gamma,delta,num_lam,num_calib,num_grid_hbb,ub,ub_sigma,epsilon,maxiters,bound_str)
