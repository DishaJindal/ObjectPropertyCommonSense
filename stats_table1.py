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
preprocess('table1_result', "true")
mean, std = get_mean_std('stats/atest')
properties = ['big', 'heavy', 'rigid', 'strong','fast', 'all']
print(mean)
plot([mean[x] for x in [0,3,6,9,12,15]], [std[x] for x in [0,3,6,9,12,15]], 'Test: PCE(LSTM) Properties with Accuracy',properties,'black')
plot([mean[x] for x in [1,4,7,10,13,16]], [std[x] for x in [1,4,7,10,13,16]], 'Test: PCE(Glove) Properties with Accuracy',properties,'black')
plot([mean[x] for x in [2,5,8,11,14,17]], [std[x] for x in [2,5,8,11,14,17]], 'Test: PCE(Word2Vec) Properties with Accuracy',properties,'black')
plot([mean[x] for x in [18,19,20,21,22,23]], [std[x] for x in [18,19,20,21,22,23]], 'Test: PCE(no reverse) Properties with Accuracy',properties,'black')
plot([mean[x] for x in [24,25,26,27,28,29]], [std[x] for x in [24,25,26,27,28,29]], 'Test: PCE(one-pole) Properties with Accuracy',properties,'black')
plot([mean[x] for x in [30,31,32,33,34,35]], [std[x] for x in [30,31,32,33,34,35]], 'Test: Majority Properties with Accuracy',properties,'black')

mean2, std2 = get_mean_std('stats/adev')
plot([mean2[x] for x in [0,3,6,9,12,15]], [std2[x] for x in [0,3,6,9,12,15]], 'Dev: PCE(LSTM) Properties with Accuracy',properties,'g')
plot([mean2[x] for x in [1,4,7,10,13,16]], [std2[x] for x in [1,4,7,10,13,16]], 'Dev: PCE(Glove) Properties with Accuracy',properties,'g')
plot([mean2[x] for x in [2,5,8,11,14,17]], [std2[x] for x in [2,5,8,11,14,17]], 'Dev: PCE(Word2Vec) Properties with Accuracy',properties,'g')
plot([mean2[x] for x in [18,19,20,21,22,23]], [std2[x] for x in [18,19,20,21,22,23]], 'Dev: PCE(no reverse) Properties with Accuracy',properties,'g')
plot([mean2[x] for x in [24,25,26,27,28,29]], [std2[x] for x in [24,25,26,27,28,29]], 'Dev: PCE(one-pole) Properties with Accuracy',properties,'g')
plot([mean2[x] for x in [30,31,32,33,34,35]], [std2[x] for x in [30,31,32,33,34,35]], 'Test: Majority Properties with Accuracy',properties,'g')

print(np.sum(std2)/36)
print(np.sum(std)/36)
# print([mean2[x] for x in [0,3,9,6,12,15]])
# print([mean[x] for x in [0,3,9,6,12,15]])
# print([mean2[x] for x in [1,4,10,7,13,16]])
# print([mean[x] for x in [1,4,10,7,13,16]])
# print([mean2[x] for x in [2,5,11,8,14,17]])
# print([mean[x] for x in [2,5,11,8,14,17]])

# print([mean2[x] for x in [24,25,27,26,28,29]])
# print([mean[x] for x in [24,25,27,26,28,29]])
# print([mean2[x] for x in [18,19,21,20,22,23]])
# print([mean[x] for x in [18,19,21,20,22,23]])

# print([mean[x] for x in [30,31,32,33,34,35]])
# print([mean2[x] for x in [30,31,32,33,34,35]])