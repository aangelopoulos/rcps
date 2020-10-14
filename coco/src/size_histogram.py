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
from recall_histogram import get_tlambda 

import pdb

parser = argparse.ArgumentParser(description='ASL MS-COCO predictor')

parser.add_argument('--model_path',type=str,default='../models/MS_COCO_TResNet_xl_640_88.4.pth')
parser.add_argument('--dset_path',type=str,default='../data/')
parser.add_argument('--model_name',type=str,default='tresnet_xl')
parser.add_argument('--input_size',type=int,default=640)
parser.add_argument('--dataset_type',type=str,default='MS-COCO')
parser.add_argument('--batch_size',type=int,default=5000)
parser.add_argument('--th',type=float,default=0.7)

def get_lamhat_precomputed(dataloader, gamma, delta, num_lam, num_val, tlambda):
    lams = torch.linspace(0,1,num_lam)
    tlam = torch.zeros_like(lams)
    Rhatlam = torch.zeros_like(lams)
    for i in range(lams.shape[0]):
        lam = lams[i]
        def _label_estimator(x):
            return (x > lam).to(float)
        prec, rec, sz = validate_precomputed(dataloader, _label_estimator, nStop=num_val, print_bool=False)
        Rhatlam[i] = 1-rec
        tlam[i] = tlambda(gamma-Rhatlam[i],delta) 

    lamhat_idxs = np.argwhere(Rhatlam+tlam < gamma)
    if lamhat_idxs.shape[0] == 0:
        return torch.Tensor([0.0])
    lamhat_idx = lamhat_idxs[-1]
    if lamhat_idx.shape[0] == 0:
        return torch.Tensor([0.0])
    return lams[lamhat_idx[-1]] 

def trial_precomputed(dataset,gamma,delta,num_lam,num_val,batch_size,tlambda):
    total = len(dataset)
    calib_data, val_data = torch.utils.data.random_split(dataset, [num_val,total-num_val])
    calib_dataloader = torch.utils.data.DataLoader(calib_data,batch_size=batch_size,shuffle=False)
    val_dataloader = torch.utils.data.DataLoader(val_data,batch_size=batch_size,shuffle=False)

    lhat = get_lamhat_precomputed(calib_dataloader, gamma, delta, num_lam, num_val, tlambda)
    def label_estimator(x):
        return (x > lhat).to(float)

    for scores, labels in val_dataloader:
        est_labels = label_estimator(scores)

    return est_labels.sum(dim=1) 

def plot_sizes(sizes,gamma,delta):
    sns.displot(data=sizes)
    plt.xlabel('sizes')
    plt.savefig(f'../outputs/histograms/{gamma}_{delta}_size_histogram.pdf')

def experiment(gamma,delta,num_lam,num_val,epsilon,num_trials):
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

    # get the precomputed binary search
    tlambda = get_tlambda()

    sizes = torch.Tensor([]).to(float) 
    for i in tqdm(range(num_trials)):
        sizes = torch.cat((sizes, trial_precomputed(dataset_precomputed,gamma,delta,num_lam,num_val,args.batch_size,tlambda)),dim=0)

    plot_sizes(sizes.numpy(),gamma,delta)


if __name__ == "__main__":
    with torch.no_grad():
        sns.set(palette='pastel',font='serif')
        sns.set_style('white')
        fix_randomness(seed=0)
        args = parse_args(parser)

        gammas = [0.05,0.1]
        deltas = [0.1,0.1]
        params = list(zip(gammas,deltas))
        num_lam = 100 
        num_val = 4000 
        epsilon = 0.01
        num_trials = 10 
        for gamma, delta in params:
            print(f"\n\n\n ============           NEW EXPERIMENT gamma={gamma} delta={delta}           ============ \n\n\n") 
            experiment(gamma,delta,num_lam,num_val,epsilon,num_trials)
         
        #estimate
        #def label_estimator(x):
        #    return (x > 0.857).to(float)
        #validate_precomputed(torch.utils.data.DataLoader(dataset_precomputed,batch_size=args.batch_size,shuffle=True), label_estimator, nStop=100000, print_bool=True)
        #pdb.set_trace()
        #mAP = get_mAP(dataloader, model, corr, 20, args.batch_size, True)
        #print(f"mAP: {mAP}")

#        def label_estimator(x):
#            return (x > 0.857).to(float)

#        prec, rec, sz = validate(dataloader,model,label_estimator,corr,nStop=100000,print_bool=True) 
