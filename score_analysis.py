import torch
import numpy as np
import pickle
import random
import copy
from IPython.core.debugger import set_trace
from torch.autograd import Variable
import argparse
import time
import math
import torch.nn as nn
import os
import matplotlib.pyplot as plt 

# Importing other modules
from datautil import Dictionary, DataLoader
from models import Emb_Similarity, Model
from train_evaluate import *

def plot(objects, list1, list2, title):
	for i,txt in enumerate(objects):
	    x1 = list1[i]
	    x2 = list2[i]
	    y1 = 0
	    y2 = 2
	    plt.scatter(x1, y1, marker='x', color='red')
	    plt.text(x1+0.05, y1+0.05, txt, fontsize=9)
	    plt.scatter(x2, y2, marker='x', color='g')
	    plt.text(x2+0.05, y2+0.05, txt, fontsize=9)
	plt.title(title)
	plt.tight_layout()
	plt.savefig('figs/' + title + '.png')
	plt.clf()

use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")

parser = argparse.ArgumentParser(description='ObjectPropertyCommonSense')
parser.add_argument('--data', type=str, default='data/')
parser.add_argument('--test', type=str, default= "verb_physics_test_5")
parser.add_argument('--devtest', type=str, default="true")
parser.add_argument('--dev', type=str, default= "verb_physics_dev_5")
parser.add_argument('--train', type=str, default= "verb_physics_train_5")
parser.add_argument('--embtype', type=str, default="lstm")
parser.add_argument('--embeddingSize', type=int, default=1024)
args = parser.parse_args()

location = args.data + args.embtype + "/" + args.embtype+'.6B.' + str(args.embeddingSize) + 'd'+ '-weights-norm' + '.refined.npy'
embedding = torch.FloatTensor(np.load(location))

# Loading Dictionary
dict = Dictionary(args.data, args.embtype)
corpus = DataLoader(dict, args.data, args.embtype, args.train, args.devtest, args.dev, args.test)

# Model-PCE(3 way)
model = Model(embedding, args.embeddingSize)
training_data = corpus.train
train(model, training_data, 800, "false", "false")

R1 = dict.get_idx('big')
R2 = dict.get_idx('small')
big_list = []

R3 = dict.get_idx('heavy')
R4 = dict.get_idx('light')
heavy_list = []

objects = ['tiger', 'zebra', 'rat', 'wasp','pigeon','mouse']
for obj in objects:
  X = dict.get_idx(obj)
  if X is not None:
    big_list.append(model.score(X, R1, R2).item())
    heavy_list.append(model.score(X, R3, R4).item())
    # print("Score: %f", model.score(X, R1, R2))
    
plot(objects, big_list, heavy_list, "Relative Sizes and Weights")