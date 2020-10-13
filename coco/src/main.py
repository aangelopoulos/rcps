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
from utils import *
import pdb

parser = argparse.ArgumentParser(description='ASL MS-COCO predictor')

parser.add_argument('--model_path',type=str,default='../models/MS_COCO_TResNet_xl_640_88.4.pth')
parser.add_argument('--dset_path',type=str,default='../data/')
parser.add_argument('--model_name',type=str,default='tresnet_xl')
parser.add_argument('--input_size',type=int,default=640)
parser.add_argument('--dataset_type',type=str,default='MS-COCO')
parser.add_argument('--batch_size',type=int,default=64)
parser.add_argument('--th',type=float,default=0.7)

if __name__ == "__main__":
    with torch.no_grad():
        args = parse_args(parser)

        dataset = tv.datasets.CocoDetection('../data/val2017','../data/annotations_trainval2017/instances_val2017.json',transform=tv.transforms.Compose([tv.transforms.Resize((args.input_size, args.input_size)),
                                                                                                                                                         tv.transforms.ToTensor()]))
        dataloader = torch.utils.data.DataLoader(dataset,batch_size=args.batch_size,shuffle=False)
        print('Dataset Loaded')

        state = torch.load('../models/MS_COCO_TResNet_xl_640_88.4.pth', map_location='cpu')
        classes_list = np.array(list(state['idx_to_class'].values()))
        args.num_classes = state['num_classes']
        model = create_model(args).cuda()
        model.load_state_dict(state['model'], strict=True)
        model.eval()
        print('Model Loaded')
        corr = get_correspondence(classes_list,dataset.coco.cats)

        mAP = get_mAP(dataloader, model, corr, 20, args.batch_size, True)
        print(f"mAP: {mAP}")

        #def label_estimator(x):
        #    tw = torch.where(x >= args.th)
        #    est_labels = [torch.Tensor([])]*args.batch_size
        #    for k in tw[0].unique():
        #        est_labels[k] = tw[1][tw[0]==k]
        #    return est_labels

        #prec, rec, sz = validate(dataloader,model,label_estimator,corr,True) 
