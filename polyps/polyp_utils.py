import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
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
from scipy.optimize import brentq
from skimage.transform import resize
from skimage.morphology import label as connected_components
from skimage.morphology import area_opening
from skimage.morphology import area_closing
from skimage.morphology import white_tophat, disk
from skimage.morphology.extrema import h_maxima
from skimage.morphology import reconstruction
from skimage.filters import gaussian
from skimage.measure import regionprops
import seaborn as sns
import time
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
        regions = np.load(cache_path + 'regions.npy')
        num_components = np.load(cache_path + 'num_components.npy')
        print(f'Loaded {sigmoids.shape[0]} labeled examples from cache.')
    except:
        num_examples = get_num_examples(folders)
        print(f'Caching {num_examples} labeled examples.')
        img_names = ['']*num_examples
        sigmoids = np.zeros((num_examples,test_size, test_size))
        masks = np.zeros((num_examples,   test_size, test_size))
        regions = np.zeros((num_examples, test_size, test_size))
        num_components = np.zeros((num_examples,))
        
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
                temp_mask = resize(gt, (test_size, test_size), anti_aliasing=False)
                #temp_mask = area_opening(temp_mask, area_threshold=10, connectivity=2)
                #temp_mask = area_closing(temp_mask, area_threshold=10, connectivity=2)
                temp_mask = gaussian(temp_mask, sigma=1, multichannel=True)
                temp_mask[temp_mask > 0.5] = 1
                masks[k,:,:], num_components[k] = connected_components(temp_mask, background=0, return_num=True, connectivity=2)
                regions[k,:,:] = find_peaks(sigmoids[k,:,:])
                k += 1
        np.save(cache_path + 'sigmoids', sigmoids)
        np.save(cache_path + 'img_names', img_names)
        np.save(cache_path + 'masks', masks)
        np.save(cache_path + 'regions', regions)
        np.save(cache_path + 'num_components', num_components)
    return np.array(img_names), torch.tensor(sigmoids), torch.tensor(masks), torch.tensor(regions), torch.tensor(num_components)

def empirical_risk_01(T, risk_mass, masks): # lambda in [-1,0]
    result = (masks - T.to(int)) # as lambda grows, the sets grow.
    F.relu(result, inplace=True) 
    result = result.to(float).sum(dim=1).sum(dim=1)/masks.sum(dim=1).sum(dim=1) # Normalize by the size of the tumor.
    return result.mean().item(), result.std().item() #first and second moments needed for some bounds 
def risk_mass_01(sigmoids):
    return sigmoids

def get_lambda_hat_clt_01(sigmoids, masks, gamma, delta, num_lam, lam_lim):
    risk_mass = risk_mass_01(sigmoids)
    def _condition(lam):
        Tlam = risk_mass >= -lam
        Rhat, sigmahat = empirical_risk_01(Tlam, risk_mass, masks)
        t = -norm.ppf(delta)*sigmahat/np.sqrt(sigmoids.shape[0]) 
        return Rhat + t - gamma
    return brentq(_condition, -0.01, -0.99, xtol=1e-3, rtol=1e-3) 

def get_lambda_hat_conformal(sigmoids, masks, gamma, delta):
    risk_mass = risk_mass_01(sigmoids)
    pdb.set_trace()
    lowest_risk_mass_per_polyp = np.array([(sigmoids[i][masks[i]==1]).min() for i in range(masks.shape[0])])
    return -np.quantile(lowest_risk_mass_per_polyp, delta)

# TODO: Rewrite this with brentq
#def get_lambda_hat_01(sigmoids, masks, gamma, delta, num_lam, lam_lim):
#    lams = torch.linspace(lam_lim[0], lam_lim[1], num_lam)
#    lams = torch.flip(lams,(0,)) # starting from the largest values of lambda, then finding the first non-valid one.
#    lam = None
#    Tlam = np.zeros_like(sigmoids).astype(bool)
#    for i in range(lams.shape[0]):
#        lam = lams[i]
#        risk_mass = risk_mass_01(Tlam, sigmoids)
#        Tlam = risk_mass >= -lam
#        Rhat, sigmahat = empirical_risk_01(Tlam, risk_mass, masks)
#        t = -norm.ppf(delta)*sigmahat/np.sqrt(sigmoids.shape[0]) 
#        print(f'\r lambda:{lam:.3f}, gamma:{gamma}, Rhat+t: {(Rhat+t):.3f}', end='')
#        if Rhat > gamma:
#            break
#        if Rhat + t > gamma:
#            break 
#            if i > 0:
#                lam = lams[i-1] # deal with the edge case where you went slightly too far.
#    print('')
#    return lam.item()

def risk_mass_l2(T, sigmoids):
    ones = np.where(T)
    zeros = np.where(~T)
    B = np.sqrt((sigmoids.shape[1]**2) + (sigmoids.shape[2]**2)) # Max risk is length of diagonal in pixels
    risk_mass = (sigmoids * (~T).to(int)).float() # Risk zero for things in the set
    # Deal with special cases where things are empty
    if ones[0].shape[0] == 0:
        return risk_mass
    if zeros[0].shape[0] == 0:
        return risk_mass

    #ones = [x[:,None] for x in ones] # For later concatenation
    #zeros = [x[:,None] for x in zeros] 
    # For each image
    #print('')
    faiss_index_list = get_faiss_index_list(sigmoids.shape[0])
    for i in range(T.shape[0]):
        ones_i = ones[0]==i
        if ones_i.sum()==0:
            continue
        rows_1 = ones[1][ones_i,None].astype(np.float32)
        cols_1 = ones[2][ones_i,None].astype(np.float32)
        start = time.time()
        faiss_index_list[i].add(np.concatenate((rows_1,cols_1),axis=1)) # Add all the 1s from image i to the index. (TODO: Manually check redundancy?)
         
        zeros_i = zeros[0]==i
        if zeros_i.sum()==0:
            continue
        rows_0 = zeros[1][zeros_i]
        cols_0 = zeros[2][zeros_i]

        add = time.time()
        l2, I = faiss_index_list[i].search(np.concatenate((rows_0[:,None].astype(np.float32),cols_0[:,None].astype(np.float32)),axis=1), 1)

        l2 = torch.tensor(l2).squeeze().sqrt()/B
        l2[l2>0.1] = 0.1

        search = time.time()
        dummy = i*torch.ones_like(torch.tensor(rows_0))
        risk_mass[(dummy,torch.tensor(rows_0),torch.tensor(cols_0))] *= l2 # Fraction of the image diagonal.
        multiply = time.time()
        #print(f"i:{i}, add: {add-start:.3f}, search: {search-add:.3f}, multiply: {multiply-search:.3f}") 
        #risk_mass[i].index_put_((rows,cols),torch.tensor(l2.astype(np.float32)) * risk_mass[i][(rows, cols)])
        #risk_mass[i][np.ix_(rows,cols)] *= torch.tensor(l2) #TODO: THIS IS PROBABLY INCORRECT AS WRITTEN
    #print('')
             
    return risk_mass

def empirical_risk_l2(T, risk_mass, masks):
    missed = (masks - T.to(int)) # as lambda grows, the sets grow.
    F.relu(missed, inplace=True) 
    missed = (missed.to(float) * risk_mass).sum(dim=1).sum(dim=1)/masks.sum(dim=1).sum(dim=1) # TODO: Normalize here.
    return missed.mean().item(), missed.std().item() #first and second moments needed for some bounds 

def get_faiss_index_list(n):
    cfg = faiss.GpuIndexFlatConfig()
    cfg.useFloat16 = False
    cfg.device = 0 

    flat_config = [cfg]
    resources = [faiss.StandardGpuResources()]

    faiss_index_list = [faiss.GpuIndexFlatL2(resources[0], 2, flat_config[0]) for i in range(n)] # We need a nearest neighbor's call for every image.
    return faiss_index_list

def construct_set_l2(sigmoids, dzeta, lam):
    zeta = 1
    Tlam = torch.zeros_like(sigmoids).to(bool)

    while zeta >= -lam:
        zeta = zeta - dzeta
        risk_mass = risk_mass_l2(Tlam, sigmoids)
        Tlam = Tlam + (risk_mass >= zeta) # This is a logical or operation. Add to the set!
    return Tlam

def get_lambda_hat_clt_l2(sigmoids, masks, gamma, delta, num_lam, lam_lim):
    lams = torch.linspace(lam_lim[0], lam_lim[1], num_lam)
    lams = torch.flip(lams,(0,)) # starting from the largest values of lambda, then finding the first non-valid one.
    lam = None

    for i in range(lams.shape[0]):
        lam = lams[i]
        Tlam = construct_set_l2(sigmoids, abs(lam_lim[0]-lam_lim[1])/num_lam, lam)
        risk_mass = risk_mass_l2(Tlam, sigmoids)
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

def calib_test_split(list_arraylike, num_calib):
    total = np.array(list_arraylike[0]).shape[0]
    perm = torch.randperm(total)
    out = list()
    for arr in list_arraylike:
        arr = arr[perm]
        out = out + [arr[0:num_calib], arr[num_calib:]]
    return out 

#def calib_test_split(img_names, sigmoids, masks, num_components, num_calib):
#    total = sigmoids.shape[0]
#    perm = torch.randperm(sigmoids.shape[0])
#    img_names = img_names[perm]
#    sigmoids = sigmoids[perm]
#    masks = masks[perm]
#    num_components = num_components[perm]
#    calib_img_names, val_img_names = (img_names[0:num_calib], img_names[num_calib:])
#    calib_sigmoids, val_sigmoids = (sigmoids[0:num_calib], sigmoids[num_calib:])
#    calib_masks, val_masks = (masks[0:num_calib], masks[num_calib:])
#    calib_num_components, val_num_components = (num_components[0:num_calib], num_components[num_calib:])
#    return calib_img_names, val_img_names, calib_sigmoids, val_sigmoids, calib_masks, val_masks, calib_num_components, val_num_components

def fix_randomness(seed=0):
    np.random.seed(seed=seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)


"""
    Code for the local maximum version of the method.
"""
def find_peaks(sigmoid):
    sigmoid = gaussian(sigmoid,0.5)
    seed = np.copy(sigmoid)
    seed[1:-1, 1:-1] = sigmoid.min()
    mask = sigmoid
    dilated = reconstruction(seed, mask, method='dilation')
    peaks = (sigmoid - dilated)
    binarized_peaks = peaks > 0.05
    labels, num_components = connected_components(binarized_peaks, background=0, return_num=True, connectivity=2)
    proposals = regionprops(labels, intensity_image=None, cache=True)
    normalization_value = np.ones_like(peaks)
    minsize = 25 
    for region in proposals:
        # take regions with large enough areas
        if region.area >= minsize:
        # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
            minr = max(minr-minsize, 0)
            minc = max(minc-minsize, 0)
            maxr = min(maxr+minsize, normalization_value.shape[0]-1)
            maxc = min(maxc+minsize, normalization_value.shape[1]-1)
            np.minimum(normalization_value[minr:maxr, minc:maxc], peaks[minr:maxr, minc:maxc].max(), out = normalization_value[minr:maxr, minc:maxc])
    peaks = np.maximum(sigmoid, peaks/normalization_value)

    return peaks

def empirical_risk_perpolyp_01(T, risk_mass, masks): # lambda in [-1,0]
    # Get the missed pixels 
    num_polyps = masks.max(dim=1)[0].max(dim=1)[0]
    missed = ((masks>0).to(int) - T.to(int)) # as lambda grows, the sets grow.
    F.relu(missed, inplace=True) 
    # Split the different polyps into different rows.
    missed = missed * masks
    results_perpolyp = torch.zeros((num_polyps.sum().int().item(), masks.shape[1], masks.shape[2]))
    masks_perpolyp = torch.zeros_like(results_perpolyp)
    k = 0
    for n in range(num_polyps.max().int().item()):
        filter_bool = (num_polyps >= n + 1) # 1, 2, 3 polyps 
        temp_missed = missed[filter_bool]
        temp_masks = masks[filter_bool]
        results_perpolyp[k:k+temp_missed.shape[0]] = (temp_missed == n + 1)
        masks_perpolyp[k:k+temp_missed.shape[0]] = (temp_masks == n+1)
        k += temp_missed.shape[0] 
    results_perpolyp = results_perpolyp.to(float).sum(dim=1).sum(dim=1)/masks_perpolyp.sum(dim=1).sum(dim=1)    
    #result = result.to(float).sum(dim=1).sum(dim=1)/masks.sum(dim=1).sum(dim=1) # Normalize by the size of the tumor.
    return results_perpolyp.mean().item(), results_perpolyp.std().item() #first and second moments needed for some bounds 

def loss_perpolyp_01(T, risk_mass, masks): # lambda in [-1,0]
    # Get the missed pixels 
    num_polyps = masks.max(dim=1)[0].max(dim=1)[0]
    missed = ((masks>0).to(int) - T.to(int)) # as lambda grows, the sets grow.
    F.relu(missed, inplace=True) 
    # Split the different polyps into different rows.
    missed = missed * masks
    results_perpolyp = torch.zeros((num_polyps.sum().int().item(), masks.shape[1], masks.shape[2]))
    masks_perpolyp = torch.zeros_like(results_perpolyp)
    k = 0
    for n in range(num_polyps.max().int().item()):
        filter_bool = (num_polyps >= n + 1) # 1, 2, 3 polyps 
        temp_missed = missed[filter_bool]
        temp_masks = masks[filter_bool]
        results_perpolyp[k:k+temp_missed.shape[0]] = (temp_missed == n + 1)
        masks_perpolyp[k:k+temp_missed.shape[0]] = (temp_masks == n+1)
        k += temp_missed.shape[0] 
    results_perpolyp = results_perpolyp.to(float).sum(dim=1).sum(dim=1)/masks_perpolyp.sum(dim=1).sum(dim=1)    
    return results_perpolyp

def risk_mass_perpolyp_01(regions):
    return regions

def get_lambda_hat_clt_perpolyp_01(regions, masks, gamma, delta, num_lam):
    risk_mass = risk_mass_perpolyp_01(regions)
    def _condition(lam):
        Tlam = risk_mass >= -lam
        Rhat, sigmahat = empirical_risk_perpolyp_01(Tlam, risk_mass, masks)
        t = -norm.ppf(delta)*sigmahat/np.sqrt(regions.shape[0]) 
        return Rhat + t - gamma
    return brentq(_condition, -0.01, -0.99, xtol=1e-3, rtol=1e-3) 

def get_lambda_hat_hbb_perpolyp_01(regions, masks, gamma, delta, num_lam, tlambda):
    lams = torch.linspace(-1,-0.29,num_lam).flip(dims=(0,))
    lam = None
    risk_mass = risk_mass_perpolyp_01(regions)
    for i in range(lams.shape[0]):
        lam = lams[i]
        est_labels = risk_mass >= -lam
        Rhat, sigmahat = empirical_risk_perpolyp_01(est_labels, risk_mass, masks)
        if Rhat >= gamma:
            break
        if Rhat + tlambda(Rhat, sigmahat, delta) >= gamma:
            break

    return lam

#def get_lambda_hat_perpolyp_01(sigmoids, masks, gamma, delta, num_lam, lam_lim):
#    lams = torch.linspace(lam_lim[0], lam_lim[1], num_lam)
#    lams = torch.flip(lams,(0,)) # starting from the largest values of lambda, then finding the first non-valid one.
#    lam = None
#    Tlam = np.zeros_like(sigmoids).astype(bool)
#    for i in range(lams.shape[0]):
#        lam = lams[i]
#        risk_mass = risk_mass_01(Tlam, sigmoids)
#        Tlam = risk_mass >= -lam
#        Rhat, sigmahat = empirical_risk_perpolyp_01(Tlam, risk_mass, masks)
#        t = -norm.ppf(delta)*sigmahat/np.sqrt(sigmoids.shape[0]) 
#        print(f'\r lambda:{lam:.3f}, gamma:{gamma}, Rhat+t: {(Rhat+t):.3f}', end='')
#        if Rhat > gamma:
#            break
#        if Rhat + t > gamma:
#            break 
#            if i > 0:
#                lam = lams[i-1] # deal with the edge case where you went slightly too far.
#    print('')
#    return lam.item()
