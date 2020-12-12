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

class Node():
    def __init__(self, id, name, sift, index, children, parents):
        self.id = id
        self.name = name
        self.sift = sift
        self.index = index
        self.children = children 
        self.parents = parents 
    
    def isRedundant(self):
        return len(self.children) == 1

    def isLeaf(self):
        return len(self.children) == 0

def getCommonParent(n1,n2):
    commonParents = [p for p in [n1] + n1.parents if p in [n2] + n2.parents]
    return commonParents[0]

def getSubtreeLeafDistance(t,l):
    p = getCommonParent(t,l)
    return ([t] + t.parents).index(p)

def dict2tree(d,root=[]):
    try:
        sift = d['sift']
    except:
        sift = ''
    try:
        index = d['index']
    except:
        index = -1
    try:
        children = d['children']
    except:
        children = []
    root = root + [ d['name'] ]
    return Node(d['id'], d['name'], sift, index, [ dict2tree(d2, root) for d2 in children ], root )

if __name__ == "__main__":
    print("tree class")
