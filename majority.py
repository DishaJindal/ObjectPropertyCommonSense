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

# Importing other modules
from datautil import Dictionary, DataLoader
from models import Emb_Similarity, Model
from train_evaluate import *

# Logging Setup
import logging
logger = logging.getLogger('pce')
fh = logging.FileHandler('pce_info.log', 'a')
logging.basicConfig(level=logging.INFO)
logger.addHandler(fh)

def evaluate_majority_class(data, testdata):
  label = data[:,5]
  cmax = -1
  count = {0:0, 1:0,2:0,3:0}
  for l in label:
    count[int(l)] = count[int(l)]+1

  for key, val in count.items():
    if(val > cmax):
      majority_prediction = key
      cmax = val
  testlabel = testdata[:,5]
  return float((testlabel.data == majority_prediction).sum()) / len(testlabel)

parser = argparse.ArgumentParser(description='PCE')
parser.add_argument('--data', type=str, default='data/')
parser.add_argument('--devtest', type=str, default="true")
parser.add_argument('--train', type=str, default= "verb_physics_train_5")
parser.add_argument('--dev', type=str, default= "verb_physics_dev_5")
parser.add_argument('--test', type=str, default= "verb_physics_test_5")
parser.add_argument('--test_relation', type=str, default="all")
args = parser.parse_args()
logger.debug(args)

location = args.data + 'lstm/lstm.6B.1024d'+ '-weights-norm' + '.refined.npy'
logger.debug("Embedding File: %s", location)
embedding = torch.FloatTensor(np.load(location))
dict = Dictionary(args.data, "lstm")
corpus = DataLoader(dict, args.data, "lstm", args.train, args.devtest, args.dev, args.test)

# Load Train and Test Data based on params
training_data = corpus.train
if args.test_relation == "all":
  test_data = corpus.test
  if args.devtest == "true":
    dev_data = corpus.dev
else:
  test_data = corpus.get_data_for_relation(corpus.test, args.test_relation, True)
  if args.devtest == "true":
    dev_data = corpus.get_data_for_relation(corpus.dev, args.test_relation, True)

if args.devtest == "true":
  logger.info("Dev Accuracy: %.2f" % evaluate_majority_class(training_data, dev_data))
logger.info("Test Accuracy: %.2f" % evaluate_majority_class(training_data, test_data))

