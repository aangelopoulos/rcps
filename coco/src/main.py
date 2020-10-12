import torch
import torchvision as tv
from asl.helper_functions.helper_functions import parse_args
from asl.loss_functions.losses import AsymmetricLoss, AsymmetricLossOptimized
from asl.models import create_model
import argparse
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import pdb

parser = argparse.ArgumentParser(description='ASL MS-COCO predictor')

parser.add_argument('--model_path',type=str,default='../models/MS_COCO_TResNet_xl_640_88.4.pth')
parser.add_argument('--dset_path',type=str,default='../data/')
parser.add_argument('--model_name',type=str,default='tresnet_xl')
parser.add_argument('--input_size',type=int,default=640)
parser.add_argument('--dataset_type',type=str,default='MS-COCO')
parser.add_argument('--th',type=float,default=None)

def get_metrics(dataset,label,est_labels,corr):
    annotations = [dataset.coco.getAnnIds(imgIds=int(x)) for x in label[0]['image_id']]
    labels = [ torch.Tensor([x['category_id'] for x in dataset.coco.loadAnns(y)]).to(int).unique() for y in annotations]
    pct = np.zeros((len(labels),))
    for i in range(len(labels)):
        correct = 0
        for j in range(len(labels[i])):
            if corr[int(labels[i][j])] in est_labels[i]:
                correct += 1
        pct[i] = correct/len(labels[i])
    return pct, pct==1.0

def get_correspondence(model_arr,dset_dict):
    corr = {}
    for i in range(model_arr.shape[0]):
        corr[i] = list(dset_dict.keys())[i]
    corr = {y:x for x,y in corr.items()}
    return corr

if __name__ == "__main__":
    with torch.no_grad():
        args = parse_args(parser)

        dataset = tv.datasets.CocoDetection('../data/val2014','../data/annotations_trainval2014/instances_val2014.json',transform=tv.transforms.Compose([tv.transforms.Resize((args.input_size, args.input_size)),
                                                                                                                                                         tv.transforms.ToTensor()]))
        dataloader = torch.utils.data.DataLoader(dataset,batch_size=64,shuffle=False)
        print('Dataset Loaded')

        state = torch.load('../models/MS_COCO_TResNet_xl_640_88.4.pth', map_location='cpu')
        classes_list = np.array(list(state['idx_to_class'].values()))
        args.num_classes = state['num_classes']
        model = create_model(args).cuda()
        model.load_state_dict(state['model'], strict=True)
        model.eval()
        print('Model Loaded')
        corr = get_correspondence(classes_list,dataset.coco.cats)

        for img, label in tqdm(dataloader):
            out = torch.sigmoid(model(img.cuda()).cpu())

            tw = torch.where(out >= args.th)
            est_labels = [tw[1][tw[0]==k] for k in tw[0].unique()]
            pct, binary = get_metrics(dataset,label,est_labels,corr)
            print(pct, binary)
