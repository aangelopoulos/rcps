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
import itertools
import seaborn as sns
import pdb

parser = argparse.ArgumentParser(description='ASL MS-COCO predictor')

parser.add_argument('--model_path',type=str,default='../models/MS_COCO_TResNet_xl_640_88.4.pth')
parser.add_argument('--dset_path',type=str,default='../data/')
parser.add_argument('--model_name',type=str,default='tresnet_xl')
parser.add_argument('--input_size',type=int,default=640)
parser.add_argument('--dataset_type',type=str,default='MS-COCO')
parser.add_argument('--batch_size',type=int,default=5000)
parser.add_argument('--th',type=float,default=0.7)

def R_to_t(R,delta,num_calib):
    return R - binom.ppf(delta/np.e,num_calib,R)/num_calib

def searchR(Rhat,delta,num_calib,epsilon):
    def _condition(R):
        return R - R_to_t(R,delta,num_calib) < Rhat 

    lbR = 0
    ubR = 1
    R = 0.5
    # Returns the biggest R consistent with _condition
    while ubR-lbR > epsilon:
        if _condition(R):
            lbR = R
            R = (ubR+lbR)/2
        else:
            ubR = R
            R = (ubR+lbR)/2
    lbR = max(lbR,epsilon)
    return lbR

# Returns tlambda table 
def get_tlambda(npts,num_calib,ub=0.2):
    tlambda_fname = '../.cache/tlambda_table.pkl'
    if os.path.exists(tlambda_fname):
        tlams = pkl.load(open(tlambda_fname,'rb'))
        print("tlambda precomputed!")
    else:
        rhats = np.linspace(0,ub,npts)
        deltas = np.array([0.1,0.05,0.01,0.001]) 
        tlams = np.zeros((npts,4))
        print("computing tlambda")
        for i in tqdm(range(tlams.shape[0])):
            for j in range(tlams.shape[1]):
                R = searchR(rhats[i],deltas[j],num_calib,0.0001)
                tlams[i,j] = R_to_t(R,deltas[j],num_calib) 
        pkl.dump(tlams,open(tlambda_fname,'wb'))

    def _tlambda(rhat,delt):
        r = min(int(np.ceil(rhat/ub * npts)), npts-1)
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
        #print(f"rhat:{rhat.item()}, g:{g},tlam:{tlams[g,d]}")
        return tlams[r,d]

    return _tlambda

if __name__ == "__main__":
    with torch.no_grad():
        ps = [0.05, 0.1, 0.2]
        deltas = [0.01, 0.05, 0.1, 0.2]
        params = list(itertools.product(deltas,ps))
        num_lam = 1000 
        num_calib = 4000 
        epsilon = 0.0001
        num_trials = 50000 
        tlambda = get_tlambda(num_lam,num_calib)
        for delta, p in params:
            print(f"\n\n\n ============           NEW EXPERIMENT delta={delta}           ============ \n\n\n") 
            Rhat = np.random.binomial(num_calib,p,size=(num_trials,))/num_calib
            upper_bound = np.zeros_like(Rhat)
            for i in tqdm(range(num_trials)):
                upper_bound[i] = Rhat[i] + tlambda(Rhat[i],delta)
            print(f"Risk: {1-(upper_bound>p).mean()}, Theory: {delta/np.e}")

