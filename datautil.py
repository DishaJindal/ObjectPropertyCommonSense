import torch
import numpy as np
import pickle
import random
import copy
from IPython.core.debugger import set_trace
from torch.autograd import Variable

class Dictionary(object):
  def __init__(self, path, embtype):
    self.word2idx = pickle.load(open(path + embtype + "/" + embtype + '.6B.vocab.refined.pickle','rb'))

  def get_idx(self, word):
    if word in self.word2idx:
      return self.word2idx[word]

class DataLoader(object):
  def __init__(self, vocab, path, embtype, train_data, dev_test, dev_data, test_data):
    self.dictionary = vocab
    self.train = self.preprocess(pickle.load(open(path +train_data+'.pickle','rb')))
    if dev_test == "true":
      self.dev = self.preprocess(pickle.load(open(path +dev_data+'.pickle','rb')))
      self.reverse_dev = self.preprocess(pickle.load(open(path +dev_data+'.pickle','rb')), True)
    self.test = self.preprocess(pickle.load(open(path +test_data+'.pickle','rb')))
    self.reverse_test = self.preprocess(pickle.load(open(path +test_data+'.pickle','rb')), True)
  
  def get_data_for_relation(self, data, relation, match):
    id =  int(self.dictionary.word2idx[relation])
    filtered_data = torch.LongTensor(data.shape[0], 6).zero_()
    count = 0
    for row in data:
      if match:    
        if row[4].item() == id:
          filtered_data[count] = row
          count = count + 1
      else:
        if row[4].item() != id:
          filtered_data[count] = row
          count = count + 1
    return filtered_data[:count,:]
  
  def preprocess(self, data, reverse = False):
    relabelmap = {-1:0, 0:1, 1:2, -42:3}
    formatted_data = torch.LongTensor(len(data),6).zero_()
    valid_data = self.filter_data_not_in_dictionary(data, relabelmap)
    
    count = 0
    relation_map = {"tall":"short", "expensive": "cheap", "dense" : "light", "mobile":"immobile", "heroic": "villainous", "dangerous":"safe", "round":"squarish", "liberal":"conservative", "shapeless":"shaped", "compressible": "incompressible", "dry" : "wet", "long":"brief", "delicious":"tasteless", "fury" : "furless", "loud" : "quiet", "sharp":"dull", "bright":"dark", "viscous":"watery", "social" : "solitary", "intelligent":"stupid", "hot":"cold", "rough":"smooth","aerodynamic":"clumsy", "healthy":"unhealthy", "thick":"thin", "northern":"southern","western":"eastern","big":"small","heavy":"light","strong":"breakable","rigid":"flexible","fast":"slow" }
    
    for row in valid_data:
      formatted_data[count, 2] = self.dictionary.word2idx[relation_map[row[2]]]
      formatted_data[count, 3] = self.dictionary.word2idx["similar"]
      formatted_data[count, 4] = self.dictionary.word2idx[row[2]]
      formatted_data[count, 5] = relabelmap[row[3]]
      if not reverse:
        formatted_data[count, 0] = self.dictionary.word2idx[row[0]]
        formatted_data[count, 1] = self.dictionary.word2idx[row[1]]
      else:
        formatted_data[count, 0] = self.dictionary.word2idx[row[1]]
        formatted_data[count, 1] = self.dictionary.word2idx[row[0]]
      count = count + 1
     
    return formatted_data
  
  def poleSentivityProcess(self, data, word1, word2):
    for row in data:
      row[2] = self.dictionary.word2idx[word1]
      row[4] = self.dictionary.word2idx[word2]
    return data
  
  def filter_data_not_in_dictionary(self, data, relabelmap):
      filtered_data = []
      for row in data:
        if row[0] in self.dictionary.word2idx and row[1] in self.dictionary.word2idx and row[3] in relabelmap:
          filtered_data.append(row)
      return filtered_data

