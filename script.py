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
# import sys; sys.argv=['']; del sys

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

use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")

parser = argparse.ArgumentParser(description='ObjectPropertyCommonSense')
parser.add_argument('--data', type=str, default='data/')
parser.add_argument('--test', type=str, default= "verb_physics_test_5")
parser.add_argument('--devtest', type=str, default="true")
parser.add_argument('--dev', type=str, default= "verb_physics_dev_5")
parser.add_argument('--train', type=str, default= "verb_physics_train_5")
parser.add_argument('--embtype', type=str, default="glove")
parser.add_argument('--embeddingSize', type=int, default=300)
parser.add_argument('--test_relation', type=str, default="all")
parser.add_argument('--zero', type=str, default="false")
parser.add_argument('--reverse', type=str, default="true")
parser.add_argument('--poleSensitivity', type=str, default="false")
parser.add_argument('--poleWord1', type=str, default="slow")
parser.add_argument('--poleWord2', type=str, default="speedy")
parser.add_argument('--onepole', type=str, default="false")
parser.add_argument('--four', type=str, default="true")
args = parser.parse_args()
logger.debug("########################################")
logger.info(args)

# Loading Embeddings
#args.data = "gdrive/"+'My Drive/' + args.data
location = args.data + args.embtype + "/" + args.embtype+'.6B.' + str(args.embeddingSize) + 'd'+ '-weights-norm' + '.refined.npy'
logger.debug("Embedding File: %s", location)
embedding = torch.FloatTensor(np.load(location))

# Loading Dictionary
dict = Dictionary(args.data, args.embtype)
corpus = DataLoader(dict, args.data, args.embtype, args.train, args.devtest, args.dev, args.test)
# corpus = DataLoader(dict, args.data, args.embtype, args.train, args.test)

# Model
model = Model(embedding, args.embeddingSize)
model = model.to(device)

# Load Train and Test Data based on params
if args.zero == "true":
  training_data = corpus.get_data_for_relation(corpus.train, args.test_relation, False)
else:
  training_data = corpus.train

if args.test_relation == "all":
  test_data = corpus.test
  reverse_test_data = corpus.reverse_test
  if args.devtest == "true":
    dev_data = corpus.dev
    reverse_dev_data = corpus.reverse_dev
else:
  test_data = corpus.get_data_for_relation(corpus.test, args.test_relation, True)
  reverse_test_data = corpus.get_data_for_relation(corpus.reverse_test, args.test_relation, True)
  if args.devtest == "true":
    dev_data = corpus.get_data_for_relation(corpus.dev, args.test_relation, True)
    reverse_dev_data = corpus.get_data_for_relation(corpus.reverse_dev, args.test_relation, True)

if args.poleSensitivity == "true":
  test_data = corpus.poleSentivityProcess(test_data, args.poleWord1, args.poleWord2)
  reverse_test_data = corpus.poleSentivityProcess(test_data, args.poleWord1, args.poleWord2)
  if args.devtest == "true":
    dev_data = corpus.poleSentivityProcess(dev_data, args.poleWord1, args.poleWord2)
    reverse_dev_data = corpus.poleSentivityProcess(reverse_dev_data, args.poleWord1, args.poleWord2)
    
logger.debug("Total Training Data: %s; Filtered Training Data: %s", corpus.train.shape, training_data.shape)
logger.debug("Total Test Data: %s; Filtered Test Data: %s", corpus.test.shape, test_data.shape)

# Train Model
train(model, training_data, 800, args.onepole, args.four)

# Evaluate Model
if args.devtest == "true":
  logger.debug("Total Dev Data: %s; Filtered Dev Data: %s", corpus.dev.shape, dev_data.shape)
  logger.info("Dev Accuracy: %.2f" % evaluate(model, dev_data, reverse_dev_data, args.reverse, args.onepole, args.four))
logger.info("Test Accuracy: %.2f" % evaluate(model, test_data, reverse_test_data, args.reverse, args.onepole, args.four))

