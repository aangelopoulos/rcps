import faiss
import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
import imageio as io
import matplotlib.pyplot as plt
import pandas as pd
from PraNet.lib.PraNet_Res2Net import PraNet
from PraNet.utils.dataloader import test_dataset
import pathlib
import random
from scipy.stats import norm
from skimage.transform import resize
import seaborn as sns
from tqdm import tqdm
import pdb

def get_num_examples(folders):
    num = 0
    for folder in folders:
        num += len([name for name in os.listdir(folder + '/images/')])
    return num

def get_data(cache_path):
    model_path = './PraNet/snapshots/PraNet_Res2Net/PraNet-19.pth'
    test_size = 352
    T = 10 
    folders = ['HyperKvasir', 'CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']
    folders = ['./PraNet/data/TestDataset/' + x for x in folders]

    try:
        img_names = np.load(cache_path + 'img_names.npy')
        sigmoids = np.load(cache_path + 'sigmoids.npy')
        masks = np.load(cache_path + 'masks.npy')
        print(f'Loaded {sigmoids.shape[0]} labeled examples from cache.')
    except:
        num_examples = get_num_examples(folders)
        print(f'Caching {num_examples} labeled examples.')
        img_names = ['']*num_examples
        sigmoids = np.zeros((num_examples,test_size, test_size))
        masks = np.zeros((num_examples,   test_size, test_size))
        
        k = 0

        for data_path in folders:
            model = PraNet()
            model.load_state_dict(torch.load(model_path))
            model.cuda()
            model.eval()

            os.makedirs(cache_path, exist_ok=True)
            image_root = '{}/images/'.format(data_path)
            gt_root = '{}/masks/'.format(data_path)
            test_loader = test_dataset(image_root, gt_root, test_size)

            for i in range(test_loader.size):
                image, gt, name = test_loader.load_data()
                print(f"\33[2K\r Processing {name}", end="")
                gt = np.asarray(gt, np.float32)
                gt /= (gt.max() + 1e-8)
                image = image.cuda()

                res5, res4, res3, res2 = model(image)
                
                # Populate the arrays
                img_names[k] = image_root + '/' + name
                sigmoids[k,:,:] = (res2/T).sigmoid().detach().cpu().numpy()
                masks[k,:,:] = resize(gt, (test_size, test_size), anti_aliasing=False) 
                k += 1
        np.save(cache_path + 'sigmoids', sigmoids)
        np.save(cache_path + 'img_names', img_names)
        np.save(cache_path + 'masks', masks)
    return img_names, torch.tensor(sigmoids), torch.tensor(masks)

def empirical_risk_01(T, risk_mass, masks): # lambda in [-1,0]
    result = (masks - T.to(int)) # as lambda grows, the sets grow.
    F.relu(result, inplace=True) 
    result = result.to(float).sum(dim=1).sum(dim=1)/masks.sum(dim=1).sum(dim=1) # Normalize by the size of the tumor.
    return result.mean().item(), result.std().item() #first and second moments needed for some bounds 

def risk_mass_01(T, sigmoids):
    return sigmoids

# TODO: Rewrite this with brentq
def get_lambda_hat_01(sigmoids, masks, gamma, delta, num_lam, lam_lim):
    lams = torch.linspace(lam_lim[0], lam_lim[1], num_lam)
    lams = torch.flip(lams,(0,)) # starting from the largest values of lambda, then finding the first non-valid one.
    lam = None
    Tlam = np.zeros_like(sigmoids).astype(bool)
    for i in range(lams.shape[0]):
        lam = lams[i]
        risk_mass = risk_mass_01(Tlam, sigmoids)
        Tlam = risk_mass >= -lam
        Rhat, sigmahat = empirical_risk_01(Tlam, risk_mass, masks)
        t = -norm.ppf(delta)*sigmahat/np.sqrt(sigmoids.shape[0]) 
        print(f'\r lambda:{lam:.3f}, gamma:{gamma}, Rhat+t: {(Rhat+t):.3f}', end='')
        if Rhat > gamma:
            break
        if Rhat + t > gamma:
            break 
            if i > 0:
                lam = lams[i-1] # deal with the edge case where you went slightly too far.
    print('')
    return lam.item()

def risk_mass_l2(T, sigmoids, faiss_index_list):
    ones = np.where(T)
    zeros = np.where(~T)
    B = np.sqrt((sigmoids.shape[1]**2) + (sigmoids.shape[2]**2)) # Max risk is length of diagonal in pixels
    risk_mass = sigmoids * (~T).to(int) # Risk zero for things in the set
    # Deal with special cases where things are empty
    if ones[0].shape[0] == 0:
        risk_mass *= B 
        return risk_mass, faiss_index_list
    if zeros[0].shape[0] == 0:
        return risk_mass, faiss_index_list

    #ones = [x[:,None] for x in ones] # For later concatenation
    #zeros = [x[:,None] for x in zeros] 
    # For each image
    for i in range(T.shape[0]):
        ones_i = ones[0]==i
        rows = ones[1][ones_i,None].astype(np.float32)
        cols = ones[2][ones_i,None].astype(np.float32)
        faiss_index_list[i].add(np.concatenate((rows,cols),axis=1)) # Add all the 1s from image i to the index. (TODO: Manually check redundancy?)
         
        zeros_i = zeros[0]==i
        rows = zeros[1][zeros_i]
        cols = zeros[2][zeros_i]

        l2, I = faiss_index_list[i].search(np.concatenate((rows[:,None].astype(np.float32),cols[:,None].astype(np.float32)),axis=1), 1)
        pdb.set_trace()
        risk_mass[i][np.ix_(rows,cols)] = torch.tensor(l2) #TODO: THIS IS PROBABLY INCORRECT AS WRITTEN
             
    return risk_mass, faiss_index_list

def empirical_risk_l2(T, risk_mass, masks):
    missed = (masks - T.to(int)) # as lambda grows, the sets grow.
    F.relu(missed, inplace=True) 
    missed = (missed.to(float) * risk_mass).sum(dim=1).sum(dim=1) # TODO: Normalize here.
    return missed.mean().item(), missed.std().item() #first and second moments needed for some bounds 

def get_lambda_hat_l2(sigmoids, masks, gamma, delta, num_lam, lam_lim):
    lams = torch.linspace(lam_lim[0], lam_lim[1], num_lam)
    lams = torch.flip(lams,(0,)) # starting from the largest values of lambda, then finding the first non-valid one.
    lam = None
    Tlam = torch.zeros_like(sigmoids).to(bool)
    faiss_index_list = [faiss.IndexFlatL2(2) for i in range(Tlam.shape[0])] # We need a nearest neighbor's call for every image.
    for i in range(lams.shape[0]):
        lam = lams[i]
        risk_mass, faiss_index_list = risk_mass_l2(Tlam, sigmoids, faiss_index_list)
        Tlam = np.logical_or(Tlam, risk_mass >= -lam) # Add to the set!
        Rhat, sigmahat = empirical_risk_l2(Tlam, risk_mass, masks)
        t = -norm.ppf(delta)*sigmahat/np.sqrt(sigmoids.shape[0]) 
        print(f'\r lambda:{lam:.3f}, gamma:{gamma}, Rhat+t: {(Rhat+t):.3f}', end='')
        if Rhat > gamma:
            break
        if Rhat + t > gamma:
            break 
            if i > 0:
                lam = lams[i-1] # deal with the edge case where you went slightly too far.
    print('')
    return lam.item()

def calib_test_split(img_names, sigmoids, masks, num_calib):
    total = sigmoids.shape[0]
    perm = torch.randperm(sigmoids.shape[0])
    img_names = img_names[perm]
    sigmoids = sigmoids[perm]
    masks = masks[perm]
    calib_img_names, val_img_names = (img_names[0:num_calib], img_names[num_calib:])
    calib_sigmoids, val_sigmoids = (sigmoids[0:num_calib], sigmoids[num_calib:])
    calib_masks, val_masks = (masks[0:num_calib], masks[num_calib:])
    return calib_img_names, val_img_names, calib_sigmoids, calib_masks, val_sigmoids, val_masks

def fix_randomness(seed=0):
    np.random.seed(seed=seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
