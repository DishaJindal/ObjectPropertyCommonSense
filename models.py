import torch
import numpy as np
import pickle
import random
import copy
from IPython.core.debugger import set_trace
from torch.autograd import Variable
import torch.nn as nn

use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")

class Emb_Similarity(torch.nn.Module):
  def __init__(self, embeddings, hidden_size):
    super(Emb_Similarity, self).__init__()
    # Embedding Layer 
    self.embedding = torch.nn.Embedding(embeddings.size(0), embeddings.size(1), _weight = embeddings)
    self.embedding.weight.requires_grad = False
    
  def forward(self, input, onepole, four):
    # Extracting word embeddings
    X = self.embedding(input[:,0])
    Y = self.embedding(input[:,1])
    R1 = self.embedding(input[:,2])
    R2 = self.embedding(input[:,3])
    R3 = self.embedding(input[:,4])
    h = X-Y
    
    cos = nn.CosineSimilarity(dim=1, eps=1e-10)
    OR1 = cos(h, R1) # n
    OR2 = cos(h, R2) # n
    OR3 = cos(h, R3) # n
    output = torch.stack([OR1, OR2, OR3], 1)
    return output

class Model(torch.nn.Module):
  def __init__(self, embeddings, hidden_size):
    super(Model, self).__init__()
    # Embedding Layer 
    self.embedding = torch.nn.Embedding(embeddings.size(0), embeddings.size(1), _weight = embeddings)
    self.embedding.weight.requires_grad = False
    
    # Linear Layer with Dropout
    self.hidden_size = hidden_size
    self.linear = torch.nn.Linear(hidden_size*2, hidden_size)
    self.linear0 = torch.nn.Linear(hidden_size, hidden_size)
    self.dropout = torch.nn.Dropout(0.5)
    
    # Linear Layer for other relations
    self.linearR2 = torch.nn.Linear(hidden_size, hidden_size)
    self.linearR3 = torch.nn.Linear(hidden_size, hidden_size)


  def forward(self, input, onepole, four):
    # Extracting word embeddings
    X = self.embedding(input[:,0])
    Y = self.embedding(input[:,1])
    R1 = self.embedding(input[:,2])
    
    if onepole == "true":     
      # Deriving Embeddings for other relations 
      R2 = self.linearR2(R1)
      R2 = self.dropout(R2)
      R3 = self.linearR3(R1)
      R3 = self.dropout(R3)
    else:
      R2 = self.embedding(input[:,3])
      R3 = self.embedding(input[:,4])
     
    # Linear followed by Dropout
    h = self.linear(torch.cat([X,Y],1))
    h = self.dropout(h)
    
    # Adapting tensors to size for multiplication    
    h = torch.unsqueeze(h,1) # n*1*embsize
    R1 = torch.unsqueeze(R1,2) # n*embsize*1
    R2 = torch.unsqueeze(R2,2) # n*embsize*1
    R3 = torch.unsqueeze(R3,2) # n*embsize*1
    
    # Similarity with each relation     
    OR1 = torch.bmm(h, R1).squeeze() # n
    OR2 = torch.bmm(h, R2).squeeze() # n
    OR3 = torch.bmm(h, R3).squeeze() # n
    
    if four == "true":
      # Additional component in 4 way(R4 refers to NA)    
      hx = self.linear0(X)
      hx = self.dropout(hx)
      hx = torch.unsqueeze(hx,1) # n*1*embsize
      Ax = torch.bmm(hx, R1).squeeze() + torch.bmm(hx, R3).squeeze() # n

      hy = self.linear0(Y)
      hy = self.dropout(hy)
      hy = torch.unsqueeze(hy,1)
      Ay = torch.bmm(hy, R1).squeeze() + torch.bmm(hy, R3).squeeze() # n

      OR4 = Ax + Ay # n
      # set_trace()
      if torch.bmm(h, R1).squeeze().dim() == 0:
        dim = 0
      else:
        dim = 1
      output = torch.stack([OR1, OR2, OR3, OR4], dim)
    else:
      output = torch.stack([OR1, OR2, OR3], 1)
    return output
  
  def score(self, obj, r1, r2):
    
    X = self.embedding(torch.LongTensor([obj]).to(device))
    R1 = self.embedding(torch.LongTensor([r1]).to(device))
    R2 = self.embedding(torch.LongTensor([r2]).to(device))
    WX = self.linear(torch.cat([X,X],1))
    score = torch.mm((R1 - R2), WX.t())
    return score
