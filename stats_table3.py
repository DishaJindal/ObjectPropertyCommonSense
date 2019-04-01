import statistics 
import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
from stats import preprocess, plot, get_mean_std

if not os.path.exists('stats'):
    os.makedirs('stats')
if not os.path.exists('figs'):
    os.makedirs('figs')
preprocess('table3_result', 'false')

mean, std = get_mean_std('stats/atest')
embeddings = ['GLove', 'Word2Vec', 'LSTM']
print([mean[x] for x in [0,1,2]])
print([std[x] for x in [0,1,2]])
plot([mean[x] for x in [0,1,2]], [std[x] for x in [0,1,2]], 'Test: 4 way task: PCE dataset: Embeddings with Accuracy', embeddings,'black')
