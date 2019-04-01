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

def train(model, training_data, epochs, onepole, four):
    data = training_data[:,(0,1,2,3,4)].to(device)
    label = training_data[:,5].to(device)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters)
    for i in range(epochs):
      # Train Model
      model.train()
      output = model(data, onepole, four)

      # Loss 
      criterion = nn.CrossEntropyLoss()
      # set_trace()
      loss = criterion(output, label) 

      # Backprop
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

def evaluate(model, test_data, reverse_test_data, reverse, onepole, four):

  label = test_data[:,5].to(device)
  input = test_data[:,(0,1,2,3,4)].to(device)
  reverse_input = reverse_test_data[:,(0,1,2,3,4)].to(device)

  input.requires_grad = False
  reverse_input.requires_grad = False

  model.eval()
  output = model(input, onepole, four)

  # Average output with reverse output
  if reverse == "true":
    reverse_output = model(reverse_input, onepole, four)
    output = (output.data + torch.cat([reverse_output.data[:,(2,1,0)], reverse_output.data[:,3:]], 1))/2

  # Prediction and Accuracy
  _ , prediction = torch.max(output ,1)
  
  return float((label.data == prediction).sum()) / len(label)

def find_LC_query(model, data, onepole, four):

  input = data[:,(0,1,2,3,4)].to(device)
  input.requires_grad = False
  model.eval()
  output = torch.zeros((data.shape[0],4))
  for i in range(0,20):
    output = output + model(input, onepole, four)

  max_prediction , prediction = torch.max(output, 1)  
  least_confident = torch.argmin(max_prediction)
  # print(torch.min(max_prediction))
  # print("Data Size", data.shape)
  return least_confident.item()

def find_EMC_query(model, data, onepole, four):

  label = data[:,5].to(device)
  input = data[:,(0,1,2,3,4)].to(device)
  input.requires_grad = False
  # set_trace()
  parameters = filter(lambda p: p.requires_grad, model.parameters())
  optimizer = torch.optim.Adam(parameters)
  model.train()
  change = torch.zeros(len(data))

  for i in range(0,len(data)):
    # set_trace()
    output = model(input[i,:].unsqueeze(0), onepole, four)
    criterion = nn.CrossEntropyLoss()
    # set_trace()
    loss = criterion(output, label[i].unsqueeze(0)) 

    optimizer.zero_grad()
    loss.backward()
    # set_trace()
    change[i] = torch.sum(abs(model.linear.weight.grad)) + torch.sum(abs(model.linear0.weight.grad))
    # change = change + ((torch.sum(abs(model.linear.weight.grad),0))[0:300] + (torch.sum(abs(model.linear.weight.grad),0))[300:600] + torch.sum(abs(model.linear0.weight.grad),0))
  max_change = torch.argmax(change)
  # print("max_change: ", torch.max(change))
  return max_change.item()

def find_uncertain_synthesis_query(model, data, onepole, four):

  input = data.to(device)
  input.requires_grad = False
  model.eval()
  output = torch.zeros((data.shape[0],4))
  for i in range(0, 20):
    output = output + model(input, onepole, four)

  max_prediction , prediction = torch.max(output, 1)  
  least_confident = torch.argmin(max_prediction)
  print(torch.min(max_prediction))
  # print("Data Size", data.shape)
  return data[least_confident.item()]


