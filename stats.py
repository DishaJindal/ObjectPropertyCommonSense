import statistics 
import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess 

def preprocess(results_file, dev):
	cmd = "grep Test " + results_file + " | awk -F ':' '{print $4}' > stats/atest"
	subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
	if dev == "true":
		cmd = "grep Dev " + results_file + " | awk -F ':' '{print $4}' > stats/adev"
		subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)

def plot(mean, std, title, properties, eco):
	x_pos = np.arange(len(properties))
	fig, ax = plt.subplots()
	ax.bar(x_pos, mean, yerr=std, align='center', alpha=0.5, ecolor=eco, capsize=10)
	ax.set_ylabel('Accuracy')
	ax.set_xticks(x_pos)
	ax.set_xticklabels(properties)
	ax.set_title(title)
	ax.yaxis.grid(True)

	plt.tight_layout()
	plt.savefig('figs/'+title+'.png')
	# plt.show()

def get_mean_std(file):
	count = 0
	alist = []
	mean = []
	std = []
	for line in open(file):
		count = count +1
		alist.append(float(line.strip('\n')))
		if(count%10 == 0):
			mean.append(round(statistics.mean(alist),2))
			std.append(round(statistics.stdev(alist),4))
			alist = []
	return mean,std