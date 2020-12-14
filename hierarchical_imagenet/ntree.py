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
    commonParents = [p for p in n1.parents if p in n2.parents]
    return commonParents[-1]

def getSubtreeLeafDistance(t,l):
    p = getCommonParent(t,l)
    ls = t.parents.copy()
    ls.reverse()
    return ls.index(p)

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

def getIndexDict(t):
    myDict = {}
    if t.index >= 0:
        myDict[t.index] = t
    childDict = {}
    for child in t.children:
        childDict = {**childDict, **getIndexDict(child)}
    return {**myDict, **childDict} 

def getNameDict(t):
    myDict = {}
    myDict[t.name] = t
    childDict = {}
    for child in t.children:
        childDict = {**childDict, **getNameDict(child)}
    return {**myDict, **childDict} 

def intersection(lol): 
    l = lol[0]
    for i in range(1,len(lol)):
        l = [x for x in lol[i] if x in l]
    return l

def getSubTree(idxs,idx_dict,name_dict):
    lol = [idx_dict[i].parents for i in idxs]
    name = intersection(lol)[-1]
    return  name_dict[name]

def getMaxDepth(st,idx_dict,name_dict):
    depths = [len(getSubTree([i], idx_dict, name_dict).parents)-1 for i in range(len(idx_dict.keys()))]
    return max(depths)

if __name__ == "__main__":
    print("tree class")
