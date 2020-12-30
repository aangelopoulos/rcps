import os
import numpy as np
import torch
import pickle as pkl
import pdb

def sample_from_probs(probs, dists):
    gt = np.zeros((probs.shape[0], probs.shape[1]))
    for i in range(probs.shape[0]):
        for j in range(probs.shape[1]):
            gt[i,j] = dists[np.random.choice(probs.shape[2],p=probs[i,j,:]/np.sum(probs[i,j,:]))]
    return gt

def predict_from_probs(probs, dists):
    return dists[np.argmax(probs,axis=2)] 

def set_from_probs(probs, preds, dists, lam):
    mask = probs >= lam
    idx_null = mask.sum(axis=2)==0
    max_i = mask.shape[2] - np.argmax(mask[:,:,::-1],axis=2) - 1
    min_i = np.argmax(mask,axis=2)
    maxes = dists[max_i]
    mins = dists[min_i]
    maxes[idx_null] = preds[idx_null]
    mins[idx_null] = preds[idx_null]
    return mins, maxes

def ls_sets_from_probs(ls_probs, ls_preds, ls_dists, lam):
    return [set_from_probs(ls_probs[i], ls_preds[i], ls_dists[i], lam) for i in range(len(ls_probs))]

def arr_l2_loss(ls_sets, ls_gt):
    return np.array([l2_loss(ls_sets[i][0], ls_sets[i][1], ls_gt[i]) for i in range(len(ls_gt))])

def l2_loss(mins,maxes,gt):
    loss_mat = np.minimum( np.abs(gt-mins), np.abs(gt-maxes))
    mask = 1-((gt >= mins) & (gt <= maxes)).astype(np.float64)
    loss_mat = loss_mat * mask
    return loss_mat.mean()

def get_preds_gt(data_dir):
    ls_targets = os.listdir(data_dir) 
    prob_cache = '.cache/ls_probs.pkl'
    pred_cache = '.cache/ls_preds.pkl'
    gt_cache = '.cache/ls_gt.pkl'
    dist_cache = '.cache/dists.pkl'
    try:
        ls_probs = pkl.load( open(prob_cache,'rb') )
        ls_preds = pkl.load( open(pred_cache,'rb') )
        ls_gt = pkl.load( open(gt_cache,'rb') )
        ls_dists = pkl.load( open(dist_cache,'rb') )
    except:
        ls_probs = []
        ls_preds = []
        ls_gt = []
        ls_dists = []
        for target in ls_targets:
            if target == 'T0999':
                continue
            target_dict = pkl.load( open( data_dir + target + '/contacts/' + target + '.pickle', 'rb' ) , encoding='latin1' ) # the files were pickled with python2
            dists = np.linspace(target_dict['min_range'],target_dict['max_range'],target_dict['num_bins'])
            ls_probs = ls_probs + [target_dict['probs']]
            ls_preds = ls_preds + [predict_from_probs(target_dict['probs'], dists)]
            ls_gt = ls_gt + [sample_from_probs(target_dict['probs'],dists)]
            ls_dists = ls_dists + [dists]
        pkl.dump(ls_probs, open(prob_cache, 'wb'))
        pkl.dump(ls_preds, open(pred_cache, 'wb'))
        pkl.dump(ls_gt, open(gt_cache, 'wb'))
        pkl.dump(ls_dists, open(dist_cache, 'wb'))
    return ls_probs, ls_preds, ls_gt, ls_targets, ls_dists

if __name__ == "__main__":
    data_dir = '/scratch/aa/casp13-data/' # for now, the casp13 dataset
    ls_probs, ls_preds, ls_gt, ls_targets, ls_dists = get_preds_gt(data_dir)
    for lam in np.linspace(0,1,20):
        ls_sets = ls_sets_from_probs(ls_probs, ls_preds, ls_dists, lam)
        losses = arr_l2_loss(ls_sets, ls_gt)
        print(losses.mean())
    print("hi")
