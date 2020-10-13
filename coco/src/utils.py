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
    labels = [ torch.Tensor([x['category_id'] for x in dataset.coco.loadAnns(y)]).to(int).unique() for y in annotations]
    prec = np.zeros((len(labels),))
    rec = np.zeros((len(labels),))
    size = np.zeros((len(labels),))
    for i in range(len(labels)):
        correct = 0
        for j in range(len(labels[i])):
            try:
                if corr[int(labels[i][j])] in est_labels[i]:
                    correct += 1
            except IndexError:
                pdb.set_trace()
        rec[i] = correct/len(labels[i])
        size[i] = len(labels[i])
        if len(est_labels[i]) == 0:
            prec[i] = 1.0
        else: 
            prec[i] = correct / ( correct + len(est_labels[i]) )
    return prec, rec, size 

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
            tw = torch.where(x >= threshold)
            est_labels = [torch.Tensor([])]*batch_size
            for k in tw[0].unique():
                est_labels[k] = tw[1][tw[0]==k]
            return est_labels
        prec, rec, sz = validate(val_loader,model,label_estimator,corr,print_bool) 
        if lastRec != None:
            mAP += prec*(rec-lastRec)
        lastRec = rec
    return mAP

def validate(val_loader, model, label_estimator, corr, print_bool):
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
            if N > 100:
                break

    if print_bool:
        print('') #Endline

    return precbar.avg, recbar.avg, sizebar.avg

