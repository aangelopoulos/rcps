import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import pathlib
import os
import pickle
import random
from tqdm import tqdm
import pdb

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def get_metrics(dataset,label,est_labels,corr):
    annotations = [dataset.coco.getAnnIds(imgIds=int(x)) for x in label[0]['image_id']]
    labels = torch.zeros_like(est_labels)
    for i in range(len(annotations)):
        for annotation in dataset.coco.loadAnns(annotations[i]):
            labels[i,corr[annotation['category_id']]] = 1

    corrects = (labels * est_labels).sum(dim=1)
    sizes = est_labels.sum(dim=1)
    corrects_temp = corrects
    corrects_temp[sizes==0] = 1
    sizes_temp = sizes
    sizes_temp[sizes==0] = 1
    precs = corrects/sizes_temp
    recs = corrects/labels.sum(dim=1)
    return precs, recs, sizes 

def get_metrics_precomputed(est_labels,labels):
    corrects = (labels * est_labels).sum(dim=1)
    sizes = est_labels.sum(dim=1)
    corrects_temp = corrects
    corrects_temp[sizes==0] = 1
    sizes_temp = sizes
    sizes_temp[sizes==0] = 1
    loss = 1-corrects/labels.float().sum(dim=1) # percent correct labels
    return loss, sizes 

def get_correspondence(model_arr,dset_dict):
    corr = {}
    for i in range(model_arr.shape[0]):
        corr[i] = list(dset_dict.keys())[i]
    corr = {y:x for x,y in corr.items()}
    return corr

def get_mAP(val_loader, model, corr, num_points, batch_size, print_bool):
    thresholds = np.flip(np.linspace(0,1,num_points))
    mAP = 0
    lastRec = None 
    for threshold in thresholds:
        def label_estimator(x):
            return (x > threshold).to(float)
        prec, rec, sz = validate(val_loader,model,label_estimator,corr,print_bool) 
        if lastRec != None:
            mAP += prec*(rec-lastRec)
        lastRec = rec
    return mAP

def validate(val_loader, model, label_estimator, corr, nStop, print_bool, subset=False):
    with torch.no_grad():
        batch_time = AverageMeter('batch_time')
        precbar = AverageMeter('prec')
        recbar = AverageMeter('rec')
        sizebar = AverageMeter('size')
        # switch to evaluate mode
        model.eval()
        end = time.time()
        N = 0

        for img, label in tqdm(val_loader):
            out = torch.sigmoid(model(img.cuda()).cpu())
            if len(label)==0:
                continue
            est_labels = label_estimator(out)
            if subset:
                prec, rec, size = get_metrics(val_loader.dataset.dataset,label,est_labels,corr)
            else:
                prec, rec, size = get_metrics(val_loader.dataset,label,est_labels,corr)

            cvg = rec == 1.0

            # Update meters
            precbar.update(prec.mean(), n=prec.shape[0])
            recbar.update(rec.mean(), n=rec.shape[0])
            sizebar.update(size.mean(), n=size.shape[0])

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            N = N + size.shape[0]
            if print_bool:
                print(f'\n\rN: {N} | Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) | Prec: {precbar.val:.3f} ({precbar.avg:.3f}) | Rec: {recbar.val:.3f} ({recbar.avg:.3f}) |Size: {sizebar.val:.3f} ({sizebar.avg:.3f})\033[1A', end='')
            if N > nStop:
                break

    if print_bool:
        print('') #Endline

    return precbar.avg, recbar.avg, sizebar.avg

def validate_precomputed(val_loader, label_estimator, nStop, print_bool):
    with torch.no_grad():
        batch_time = AverageMeter('batch_time')
        precbar = AverageMeter('prec')
        recbar = AverageMeter('rec')
        sizebar = AverageMeter('size')

        end = time.time()
        N = 0

        my_iterable = tqdm(val_loader) if print_bool else val_loader
        for scores, labels in my_iterable:
            est_labels = label_estimator(scores)
            prec, rec, size = get_metrics_precomputed(est_labels,labels)

            cvg = rec == 1.0

            # Update meters
            precbar.update(prec.mean().item(), n=prec.shape[0])
            recbar.update(rec.mean().item(), n=rec.shape[0])
            sizebar.update(size.mean().item(), n=size.shape[0])
            batch_time.update(time.time() - end)

            # measure elapsed time
            N = N + size.shape[0]
            end = time.time()

            if print_bool:
                print(f'\n\rN: {N} | Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) | Prec: {precbar.val:.3f} ({precbar.avg:.3f}) | Rec: {recbar.val:.3f} ({recbar.avg:.3f}) |Size: {sizebar.val:.3f} ({sizebar.avg:.3f})\033[1A', end='')
            if N > nStop:
                break

    if print_bool:
        print('') #Endline

    return precbar.avg, recbar.avg, sizebar.avg

def validate_precomputed_oneshot(scores,labels, label_estimator):
    with torch.no_grad():
        est_labels = label_estimator(scores)
        prec, rec, size = get_metrics_precomputed(est_labels,labels)

        cvg = rec == 1.0

    return prec, rec, size 

def fix_randomness(seed):
    np.random.seed(seed=seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

# Computes logits and targets from a model and loader
def get_scores_targets(model, loader, corr):
    scores = torch.zeros((len(loader.dataset), 80))
    labels = torch.zeros((len(loader.dataset), 80))
    i = 0
    print(f'Computing logits for model (only happens once).')
    with torch.no_grad():
        for x, label in tqdm(loader):
            if len(label)==0:
                print('I encountered an unlabeled image.')
                continue
            batch_scores = torch.sigmoid(model(x.cuda())).detach().cpu()
            scores[i:(i+x.shape[0]), :] = batch_scores

            annotations = [loader.dataset.coco.getAnnIds(imgIds=int(x)) for x in label[0]['image_id']]
            batch_labels = torch.zeros((x.shape[0],80))
            for j in range(len(annotations)):
                for annotation in loader.dataset.coco.loadAnns(annotations[j]):
                    batch_labels[j,corr[annotation['category_id']]] = 1

            labels[i:(i+x.shape[0]),:] = batch_labels 
            i = i + x.shape[0]

    keep = labels.sum(dim=1) > 0
    scores = scores[keep]
    labels = labels[keep]
    
    # Construct the dataset
    dataset_logits = torch.utils.data.TensorDataset(scores, labels.long()) 
    return dataset_logits
