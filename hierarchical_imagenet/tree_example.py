import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt 
import time
from utils import * 
import torch.backends.cudnn as cudnn
import random
import json
from ntree import *

if __name__ == "__main__":
    ### Fix randomness 
    seed = 0
    np.random.seed(seed=seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    # TODO: Replace with your directory
    imagenet_val_dir = '/scratch/group/ilsvrc/val/'

    # Transform as in https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py#L92 
    transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std= [0.229, 0.224, 0.225])
                ])

    num_calib = 20000
    batch_size = 32
    # Get the conformal calibration dataset
    imagenet_calib_data, imagenet_val_data = torch.utils.data.random_split(torchvision.datasets.ImageFolder(imagenet_val_dir, transform), [num_calib,50000-num_calib])

    # Initialize loaders 
    calib_loader = torch.utils.data.DataLoader(imagenet_calib_data, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(imagenet_val_data, batch_size=batch_size, shuffle=True, pin_memory=True)

    with open('./wordnet_hierarchy.json', 'r') as file:
        data = file.read()

    imagenet_dict = json.loads(data)
    t = dict2tree(imagenet_dict)
    idx_dict = getIndexDict(t)
    name_dict = getNameDict(t)

    st = getSubTree([9, 10],idx_dict,name_dict)
    l = idx_dict[500]
    d = getSubtreeLeafDistance(st, l)
    
    pdb.set_trace()
