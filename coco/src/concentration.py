import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from core.bounds import bentkus_mu_plus
import torch
import torchvision as tv
from asl.helper_functions.helper_functions import parse_args
from asl.loss_functions.losses import AsymmetricLoss, AsymmetricLossOptimized
from asl.models import create_model
import argparse
import time
import numpy as np
from scipy.stats import binom
from scipy.optimize import brenth
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pkl
from tqdm import tqdm
from utils import *
import itertools
import seaborn as sns
from joblib import Parallel
import pdb

parser = argparse.ArgumentParser(description='ASL MS-COCO predictor')

parser.add_argument('--model_path',type=str,default='../models/MS_COCO_TResNet_xl_640_88.4.pth')
parser.add_argument('--dset_path',type=str,default='../data/')
parser.add_argument('--model_name',type=str,default='tresnet_xl')
parser.add_argument('--input_size',type=int,default=640)
parser.add_argument('--dataset_type',type=str,default='MS-COCO')
parser.add_argument('--batch_size',type=int,default=5000)
parser.add_argument('--th',type=float,default=0.7)

#def R_to_t(R,delta,num_calib):
#    return R - binom.ppf(delta/np.e,num_calib,R)/num_calib

def R_to_t(R,sigma,delta,num_calib,num_grid_hbb):
    return bentkus_mu_plus(R,sigma,num_calib,delta,num_grid_hbb) - R

def searchR(Rhat,sigmahat,delta,num_calib,num_grid_hbb,epsilon):
    def _gap(R):
        return R - R_to_t(R,sigmahat,delta,num_calib,num_grid_hbb) - Rhat

    root = brenth(_gap,0,1) 
    return max(root,epsilon)

#out = Parallel(n_jobs=-1, verbose=100)( _inner_loops_tlambda(tlams.shape[1], tlams.shape[2], rhats[i], sigmas, deltas, num_calib, num_grid_hbb) for i in range(tlams.shape[0]))
#def _inner_loops_tlambda(n_j, n_k, rhats_i, sigmas, deltas, num_calib, num_grid_hbb):
#    tlams_i = np.zeros((1,n_j,n_k)) 
#    for j in range(n_j):
#        for k in range(n_k):
#            R = searchR(rhats_i,sigmas[j],deltas[k],num_calib,num_grid_hbb,0.0001)
#            tlams_i[0,j,k] = R_to_t(R,sigmas[j],deltas[k],num_calib,num_grid_hbb) 
#    return tlams_i

# Returns tlambda table 
def get_tlambda(npts,num_calib,num_grid_hbb,ub=0.2,epsilon):
    tlambda_fname = '../.cache/tlambda_table.pkl'
    if os.path.exists(tlambda_fname):
        tlams = pkl.load(open(tlambda_fname,'rb'))
        print("tlambda precomputed!")
    else:
        # Log space this from 10-4 to 10-1.
        rhats = np.linspace(1e-10,ub,npts)
        sigmas = np.linspace(1e-10,ub,npts)
        deltas = np.array([0.1,0.05,0.01,0.001]) 
        tlams = np.zeros((npts,npts,4))
        print("computing tlambda")
        for i in tqdm(range(tlams.shape[0])):
            for j in range(tlams.shape[2]):
                #for k in range(tlams.shape[2]):
                R = searchR(rhats[i],sigmas[0],deltas[j],num_calib,num_grid_hbb,epsilon)
                tlams[i,:,j] = R_to_t(R,sigmas[0],deltas[j],num_calib,num_grid_hbb) 
        pkl.dump(tlams,open(tlambda_fname,'wb'))

    def _tlambda(rhat,sig,delt):
        r = min(int(np.floor(rhat/ub * npts)), npts-1)
        s = min(int(np.ceil(sig/ub * npts)), npts-1)
        d = None 
        if delt == 0.1:
            d = 0
        elif delt == 0.05:
            d = 1
        elif delt == 0.01:
            d = 2
        elif delt == 0.001:
            d = 3
        else:
            raise NotImplemented
        return tlams[r,s,d]

    return _tlambda

if __name__ == "__main__":
    with torch.no_grad():
        ps = [0.05, 0.1, 0.2]
        deltas = [0.001, 0.01, 0.05, 0.1]
        params = list(itertools.product(deltas,ps))
        num_lam = 1000 
        num_calib = 4000 
        num_grid_hbb = 5000
        epsilon = 0.000001
        num_trials = 1000000 
        tlambda = get_tlambda(num_lam,num_calib,num_grid_hbb,epsilon)
        for delta, p in params:
            print(f"\n\n\n ============           NEW EXPERIMENT delta={delta}, p={p}          ============ \n\n\n") 
            Rhat = np.random.binomial(num_calib,p,size=(num_trials,))/num_calib
            sigmahat = np.sqrt(Rhat*(1-Rhat)/num_calib)
            upper_bound = np.zeros_like(Rhat)
            for i in tqdm(range(num_trials)):
                upper_bound[i] = Rhat[i] + tlambda(Rhat[i],sigmahat[i],delta)
            print(f"Risk: {1-(upper_bound>p).mean()}, Theory: {delta/np.e}, Difference: {(1-(upper_bound>p).mean()-delta/np.e)/np.sqrt((delta/np.e)*(1-(delta/np.e))/num_trials)}")

