import os, sys, inspect
#sys.path.insert(1, os.path.join(sys.path[0], '../..'))
dir_path = os.path.dirname(os.path.realpath(__file__))
from core.bounds import bentkus_mu_plus, HB_mu_plus, HBB_mu_plus, WSR_mu_plus
import torch
import torchvision as tv
import argparse
import time
import numpy as np
from scipy.stats import binom
from scipy.optimize import brentq
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pkl
from tqdm import tqdm
from utils import *
import itertools
import seaborn as sns
from scipy.stats import norm
from joblib import Parallel
import pdb

def R_to_t(R,sigma,delta,num_calib,num_grid_hbb,maxiters,bound_fn):
    return bound_fn(R,sigma,num_calib,delta,num_grid_hbb,maxiters) - R

def searchR(Rhat,sigmahat,delta,num_calib,num_grid_hbb,epsilon,maxiters,bound_fn):
    def _gap(R):
        return R - R_to_t(R,sigmahat,delta,num_calib,num_grid_hbb,maxiters,bound_fn) - Rhat

    root = brentq(_gap,0,1,maxiter=maxiters) 
    return max(root,epsilon)

def get_tlambda(npts,deltas,num_calib,num_grid_hbb,ub,ub_sigma,epsilon,maxiters,bound_str,bound_fn):
    bound_str = bound_str.lower()
    tlambda_fname = dir_path + '/.cache/' + bound_str + '_' + str(npts) + '_' + str(num_calib) + '_tlambda_table.pkl'
    npts_sigma = max(int(npts/10),1)

    if bound_str == 'clt':
        def _tlambda(rhat, sig, delt):
            return -norm.ppf(delt)*sig/np.sqrt(num_calib) 
        return _tlambda

    if bound_str == 'wsr':
        # no good way to cache this :(
        def _tlambda(losses, delt): # this loss fn has different arguments
            return WSR_mu_plus(losses, delt, maxiters) - losses.mean() # R^+-Rhat = t
        return _tlambda

    if os.path.exists(tlambda_fname):
        tlams = pkl.load(open(tlambda_fname,'rb'))
        print("tlambda precomputed!")
    else:
        rhats = np.linspace(epsilon,ub,npts)
        sigmas = np.linspace(epsilon,ub_sigma,npts_sigma)
        #rhats = np.logspace(np.log10(epsilon),np.log10(ub),npts)
        #sigmas = np.logspace(np.log10(epsilon),np.log10(ub_sigma),npts_sigma)
        tlams = np.zeros((npts,sigmas.shape[0],len(deltas)))
        print("computing tlambda")

        if bound_str in ['empirical_bennett', 'hbb', 'HBB']:
            for i in tqdm(range(tlams.shape[0])):
                for j in range(tlams.shape[1]):
                    for k in range(tlams.shape[2]):
                        R = searchR(rhats[i],sigmas[j],deltas[k],num_calib,num_grid_hbb,epsilon,maxiters,bound_fn)
                        tlams[i,j,k] = R_to_t(R,sigmas[j],deltas[k],num_calib,num_grid_hbb,maxiters,bound_fn) 
        else:
            for i in tqdm(range(tlams.shape[0])):
                for k in range(tlams.shape[2]):
                    R = searchR(rhats[i],1,deltas[k],num_calib,num_grid_hbb,epsilon,maxiters,bound_fn)
                    tlams[i,:,k] = R_to_t(R,1,deltas[k],num_calib,num_grid_hbb,maxiters,bound_fn) 

        pkl.dump(tlams,open(tlambda_fname,'wb'))

    def _tlambda(rhat,sig,delt):
        r = min(int(np.floor(rhat/ub * npts)), npts-1)
        s = min(int(np.ceil(sig/ub_sigma * npts_sigma)), -1)
        #r = min(int(np.floor((np.log10(rhat)-np.log10(epsilon))/(np.log10(ub)-np.log10(epsilon)) * npts)), npts-1)
        #s = min(int(np.ceil((np.log10(sig)-np.log10(epsilon))/(np.log10(ub_sigma)-np.log10(epsilon)) * npts)), npts_sigma-1)
        d = None 
        for i in range(len(deltas)):
            if delt == deltas[i]:
                d = i
                break
        if d is None or d == None:
            raise NotImplemented

        return tlams[r,s,d]

    return _tlambda

def get_lhat_from_table(calib_loss_table, lambdas_table, gamma, delta, tlambda, bound_str):
    calib_loss_table = calib_loss_table[:,::-1]
    avg_loss = calib_loss_table.mean(axis=0)
    std_loss = calib_loss_table.std(axis=0)

    for i in range(1, len(lambdas_table)):
        Rhat = avg_loss[i]
        sigmahat = std_loss[i]
        t = tlambda(Rhat, sigmahat, delta) if bound_str not in ['WSR'] else tlambda(calib_loss_table[:,i], delta)
        if (Rhat > gamma) or (Rhat + t > gamma):
            return lambdas_table[-i+1] #TODO: i+2 ; one of the +1 comes from the overshoot of Rhat + t, and the other from 0-indexing. 

    return lambdas_table[-1]

def get_lhat_from_table_binarysearch(calib_loss_table, lambdas_table, gamma, delta, tlambda, bound_str):
    calib_loss_table = calib_loss_table[:,::-1]
    avg_loss = calib_loss_table.mean(axis=0)
    std_loss = calib_loss_table.std(axis=0)

    lb_idx = 1
    ub_idx = len(lambdas_table)-1
    i = 0 

    while lb_idx != ub_idx-1:
        i = (lb_idx + ub_idx)//2
        Rhat = avg_loss[i]
        sigmahat = std_loss[i]
        t = tlambda(Rhat, sigmahat, delta) if bound_str not in ['WSR'] else tlambda(calib_loss_table[:,i], delta)
        if (Rhat > gamma) or (Rhat + t > gamma):
            ub_idx = i
        else:
            lb_idx = i

    return lambdas_table[-i-1] #TODO: correct?

def get_lhat_conformal_from_table(calib_loss_table, lambdas_table, alpha):
    avg_loss = calib_loss_table.mean(axis=0)
    idx = np.argmax(avg_loss < alpha)
    return lambdas_table[idx]

def test_table(Rhat,delta,bound_fn):
    sigmahat = np.sqrt(2*Rhat*(1-Rhat))
    ucb1 = Rhat + tlambda(Rhat, sigmahat, delta)
    ucb2 = R_to_t(Rhat, sigmahat, delta, num_calib, num_grid_hbb, maxiters, bound_fn) + Rhat
    x1 = np.random.binomial(num_calib, ucb1, size=(num_trials,))/num_calib
    x2 = np.random.binomial(num_calib, ucb2, size=(num_trials,))/num_calib
    r1 = (x1 <= Rhat).mean() * np.e / delta
    r2 = (x2 <= Rhat).mean() * np.e / delta
    print(f"UCB fraction: {(ucb1-ucb2)/ucb2} | Table: {r1} | Direct: {r2}")

def get_bound_fn_from_string(bound_str):
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
    return bound_fn

if __name__ == "__main__":
    with torch.no_grad():
        bounds_to_plot = ['bentkus','hbb']
        ps = [0.05, 0.1, 0.2]
        deltas = [0.001, 0.01, 0.05, 0.1]
        params = list(itertools.product(deltas,ps))

        num_lam = 1500 
        num_calib = 4000 
        num_grid_hbb = 200
        epsilon = 1e-10 
        maxiters = int(1e5)
        num_trials = 100000 
        ub = 0.2
        ub_sigma = np.sqrt(2)

        for bound_str in bounds_to_plot:
            if bound_str == 'bentkus':
                bound_fn = bentkus_mu_plus
            elif bound_str == 'hbb':
                bound_fn = HBB_mu_plus

            tlambda = get_tlambda(num_lam,deltas,num_calib,num_grid_hbb,ub,ub_sigma,epsilon,maxiters,bound_str,bound_fn)

            # The test
            test_table(0.19,0.001,bound_fn)
            test_table(0.01,0.001,bound_fn)

            # Experiments
            for delta, p in params:
                print(f"\n\n\n ============      bound={bound_str} NEW EXPERIMENT delta={delta}, p={p}          ============ \n\n\n") 
                Rhat = np.random.binomial(num_calib,p,size=(num_trials,))/num_calib
                sigmahat = np.sqrt(2*Rhat*(1-Rhat))#np.sqrt(Rhat*(1-Rhat)/num_calib)
                upper_bound = np.zeros_like(Rhat)
                for i in tqdm(range(num_trials)):
                    upper_bound[i] = Rhat[i] + tlambda(Rhat[i],sigmahat[i],delta)
                e_miscoverage = (upper_bound <= p).mean()#1-(upper_bound>p).mean()
                t_miscoverage = delta/np.e # delta/np.e for bentkus, delta/2 else.
                z_value = (1-(upper_bound>p).mean()-delta/np.e)/np.sqrt((delta/np.e)*(1-(delta/np.e))/num_trials)
                print(f"Miscoverage: {e_miscoverage}, Theory: {t_miscoverage}, Miscoverage/Theory: {e_miscoverage/t_miscoverage}")

