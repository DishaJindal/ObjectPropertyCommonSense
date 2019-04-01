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
preprocess('table2_result', "true")

mean, std = get_mean_std('stats/atest')
properties = ['big', 'heavy', 'rigid', 'strong','fast']
plot([mean[x] for x in [0,1,2,3,4]], [std[x] for x in [0,1,2,3,4]], 'Test: Zero-Shot: PCE Properties with Accuracy',properties,'black')
plot([mean[x] for x in [5,6,7,8,9]], [std[x] for x in [5,6,7,8,9]], 'Test: Zero-Shot: PCE(one-pole) Properties with Accuracy',properties,'black')

mean2, std2 = get_mean_std('stats/adev')
plot([mean2[x] for x in [0,1,2,3,4]], [std2[x] for x in [0,1,2,3,4]], 'Dev: Zero-Shot: PCE Properties with Accuracy',properties,'g')
plot([mean2[x] for x in [5,6,7,8,9]], [std2[x] for x in [5,6,7,8,9]], 'Dev: Zero-Shot: PCE(one-pole) Properties with Accuracy',properties,'g')

print([mean2[x] for x in [5,6,8,7,9]])
print([mean[x] for x in [5,6,8,7,9]])
print([mean2[x] for x in [0,1,3,2,4]])
print([mean[x] for x in [0,1,3,2,4]])
