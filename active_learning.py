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

# Logging Setup
import logging
logger = logging.getLogger('pce')
fh = logging.FileHandler('pce_info.log', 'a')
logging.basicConfig(level=logging.INFO)
logger.addHandler(fh)

use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")

# Random Approach
def al_random(embedding, embeddingSize, poolsize, warmup_indices, training_data, test_data, reverse_test_data):
	model = Model(embedding, embeddingSize)
	model = model.to(device)
	random_accuracy = []
	training_indices = set(list(range(1, len(training_data)))) - set(warmup_indices)
	train(model, training_data[warmup_indices], 800, "false", "true")
	logger.info("Test Accuracy: %.2f" % evaluate(model, test_data, reverse_test_data, "true", "false", "false", "true"))
	for i in range(1, poolsize):
	  logger.info(i)
	  nextvar = random.sample(training_indices, 1)
	  warmup_indices = warmup_indices + nextvar
	  training_indices = set(training_indices) - set(nextvar)
	  train(model, training_data[warmup_indices], 20, "false", "true")
	  random_accuracy.append(evaluate(model, test_data, reverse_test_data, "true", "false", "false", "true"))
	  logger.info("Test Accuracy: %.2f" % evaluate(model, test_data, reverse_test_data, "true", "false", "false", "true"))
	logger.info(random_accuracy)
	return random_accuracy

# LC Approach
def al_lc(embedding, embeddingSize, poolsize, warmup_indices, training_data, test_data, reverse_test_data):
	model = Model(embedding, embeddingSize)
	model = model.to(device)
	lc_accuracy = []
	training_indices = list(set(list(range(1, len(training_data)))) - set(warmup_indices))
	train(model, training_data[warmup_indices], 800, "false", "true")
	logger.info("Test Accuracy: %.2f" % evaluate(model, test_data, reverse_test_data, "true", "false", "false", "true"))
	for i in range(1, poolsize):
	  logger.info(i)
	  nextvar = find_LC_query(model, training_data[training_indices], "false", "true")
	  warmup_indices = warmup_indices + [training_indices[nextvar]]
	  # set_trace()
	  training_indices.remove(training_indices[nextvar])
	  train(model, training_data[warmup_indices], 20, "false", "true")
	  acc = evaluate(model, test_data, reverse_test_data, "true", "false", "false", "true")
	  lc_accuracy.append(acc)
	  logger.info("Test Accuracy: %.2f" % acc)
	
	logger.info(lc_accuracy)
	return lc_accuracy

# EMC Approach
def al_emc(embedding, embeddingSize, poolsize, warmup_indices, training_data, test_data, reverse_test_data):
	model = Model(embedding, embeddingSize)
	model = model.to(device)
	emc_accuracy = []
	training_indices = list(set(list(range(1, len(training_data)))) - set(warmup_indices))
	train(model, training_data[warmup_indices], 800, "false", "true")
	logger.info("Test Accuracy: %.2f" % evaluate(model, test_data, reverse_test_data, "true", "false", "false", "true"))
	for i in range(1, poolsize):
	  logger.info(i)

	  nextvar = find_EMC_query(model, training_data[training_indices], "false", "true")
	  warmup_indices = warmup_indices + [training_indices[nextvar]]
	  training_indices.remove(training_indices[nextvar])

	  train(model, training_data[warmup_indices], 20, "false", "true")
	  acc = evaluate(model, test_data, reverse_test_data, "true", "false", "false", "true")
	  emc_accuracy.append(acc)
	  logger.info("Test Accuracy: %.2f" % acc)
	logger.info(emc_accuracy)
	return emc_accuracy

# Synthesis Based Approach
def al_synthesis_based(dict, embedding, embeddingSize, poolsize, warmup_indices, training_data, test_data, reverse_test_data):
	objects = ['gun', 'tomahawk', 'catapult', 'chandelier', 'minnow', 'bison', 'inn', 'mouse', 'pliers', 'birch', 'eggplant', 'brush', 'saxophone', 'trout', 'celery', 'pumpkin', 'couch', 'tack', 'screwdriver', 'ant', 'cheese', 'crowbar', 'drain', 'pier', 'harpsichord', 'bomb', 'bathtub', 'barrel', 'budgie', 'shawl', 'partridge', 'whale', 'gown', 'vest', 'shell', 'sofa', 'bench', 'canary', 'balloon', 'emu', 'housefly', 'grape', 'nightgown', 'car', 'doll', 'racquet', 'cheetah', 'axe', 'trombone', 'helmet', 'limousine', 'sack', 'chain', 'ashtray', 'butterfly', 'raft', 'rake', 'rat', 'catfish', 'duck', 'sword', 'kettle', 'cottage', 'bracelet', 'pony', 'skillet', 'fork', 'pigeon', 'socks', 'coat', 'peach', 'ball', 'swimsuit', 'chipmunk', 'peas', 'skis', 'chickadee', 'anchor', 'sheep', 'ox', 'crow', 'sandals', 'snail', 'groundhog', 'pelican', 'gorilla', 'olive', 'shack', 'freezer', 'horse', 'table', 'stone', 'pine', 'wheel', 'chair', 'dunebuggy', 'flea', 'sleigh', 'ship', 'spatula', 'violin', 'pheasant', 'otter', 'helicopter', 'onions', 'blueberry', 'mackerel', 'beans', 'grenade', 'strainer', 'raccoon', 'razor', 'bike', 'sparrow', 'honeydew', 'wasp', 'corkscrew', 'napkin', 'buzzard', 'leotards', 'dog', 'skateboard', 'bag', 'sledgehammer', 'banana', 'pickle', 'seaweed', 'porcupine', 'orange', 'chimp', 'cannon', 'belt', 'buckle', 'salamander', 'sweater', 'rocker', 'tray', 'cranberry', 'caterpillar', 'skirt', 'tuba', 'bluejay', 'accordion', 'tie', 'truck', 'yam', 'cow', 'stereo', 'door', 'bazooka', 'pie', 'platypus', 'armour', 'bowl', 'crayon', 'earmuffs', 'jar', 'rock', 'ostrich', 'cello', 'apartment', 'crab', 'mittens', 'falcon', 'iguana', 'wand', 'missile', 'bookcase', 'jeans', 'saucer', 'yacht', 'parka', 'drum', 'toilet', 'vulture', 'sailboat', 'moose', 'robin', 'seal', 'fridge', 'oak', 'hut', 'vine', 'robe', 'flute', 'taxi', 'hornet', 'pear', 'cake', 'nightingale', 'box', 'penguin', 'crown', 'biscuit', 'stork', 'mirror', 'shovel', 'pants', 'cushion', 'lime', 'rabbit', 'coyote', 'hyena', 'owl', 'doorknob', 'toy', 'dove', 'bra', 'tortoise', 'candle', 'microwave', 'banjo', 'jeep', 'elk', 'octopus', 'necklace', 'deer', 'radio', 'toaster', 'clamp', 'emerald', 'bullet', 'pajamas', 'crocodile', 'tripod', 'squid', 'salmon', 'chapel', 'paintbrush', 'wall', 'cantaloupe', 'bear', 'closet', 'tangerine', 'canoe', 'alligator', 'shield', 'eel', 'beets', 'dolphin', 'magazine', 'finch', 'sardine', 'gopher', 'clam', 'blackbird', 'clarinet', 'hawk', 'parsley', 'mat', 'lobster', 'bungalow', 'jacket', 'saddle', 'cabin', 'shirt', 'surfboard', 'baton', 'radish', 'cauliflower', 'hamster', 'harmonica', 'unicycle', 'rice', 'veil', 'typewriter', 'potato', 'walrus', 'motorcycle', 'peg', 'mink', 'trolley', 'brick', 'umbrella', 'lion', 'kite', 'scooter', 'cigarette', 'book', 'pin', 'plate', 'scissors', 'gate', 'cigar', 'drill', 'airplane', 'mug', 'faucet', 'apple', 'cedar', 'marble', 'cabbage', 'nectarine', 'lettuce', 'guppy', 'key', 'boots', 'tuna', 'cork', 'church', 'shrimp', 'cougar', 'harp', 'medal', 'raven', 'bull', 'oven', 'basement', 'corn', 'sink', 'elephant', 'guitar', 'bed', 'hoe', 'cape', 'buggy', 'clock', 'toad', 'camisole', 'squirrel', 'moth', 'python', 'knife', 'thimble', 'cabinet', 'asparagus', 'pistol', 'coconut', 'pineapple', 'beetle', 'rattle', 'bridge', 'muzzle', 'shelves', 'apron', 'tongs', 'level', 'football', 'spear', 'strawberry', 'rocket', 'tractor', 'jet', 'garage', 'carpet', 'envelope', 'comb', 'whistle', 'panther', 'mushroom', 'broom', 'microscope', 'pyramid', 'woodpecker', 'barn', 'grapefruit', 'turtle', 'sled', 'camel', 'trumpet', 'beaver', 'pearl', 'basket', 'dish', 'fence', 'ambulance', 'cherry', 'calf', 'slippers', 'cage', 'flamingo', 'prune', 'hammer', 'avocado', 'donkey', 'desk', 'curtains', 'mixer', 'pencil', 'projector', 'thermometer', 'stick', 'drapes', 'spinach', 'tent', 'frog', 'wrench', 'spade', 'scarf', 'leopard', 'skunk', 'plum', 'lemon', 'tomato', 'cucumber', 'parakeet', 'bolts', 'blender', 'crossbow', 'elevator', 'raisin', 'coin', 'telephone', 'dress', 'sandpaper', 'colander', 'harpoon', 'van', 'turnip', 'fawn', 'pen', 'bedroom', 'buffalo', 'seagull', 'bottle', 'lantern', 'cathedral', 'fox', 'shoes', 'cod', 'walnut', 'grasshopper', 'peacock', 'shed', 'submarine', 'hare', 'ladle', 'wagon', 'giraffe', 'bread', 'blouse', 'tricycle', 'turkey', 'spider', 'bayonet', 'bucket', 'carrot', 'certificate', 'lamp', 'whip', 'dagger', 'chisel', 'grater', 'stove', 'trailer', 'train', 'wheelbarrow', 'lamb', 'banner', 'pepper', 'bagpipe', 'garlic', 'willow', 'menu', 'perch', 'goldfish', 'gloves', 'mandarin', 'cart', 'rooster', 'rifle', 'piano', 'goose', 'dandelion', 'rattlesnake', 'hook', 'pig', 'rhubarb', 'swan', 'beehive', 'building', 'broccoli', 'eagle', 'cockroach', 'chicken', 'subway', 'bus', 'spoon', 'cellar', 'starling', 'bouquet', 'bureau', 'ruler', 'goat', 'cat', 'nylons', 'slingshot', 'dresser', 'worm', 'urn', 'house', 'rope', 'skyscraper', 'hatchet', 'raspberry', 'screws', 'machete', 'pillow', 'pot', 'oriole', 'shotgun', 'cloak', 'dishwasher', 'trousers', 'caribou', 'cupboard', 'revolver', 'tiger', 'zebra', 'tap', 'pan', 'escalator', 'hose', 'boat', 'zucchini', 'cup', 'cyanide', 'arsenic', 'diamond', 'gem', 'ruby', 'pearl', 'paper', 'water', 'clouds', 'thunder', 'cat', 'week', 'day', 'year', 'month', 'second', 'minute', 'hour', 'quarter', 'semester', 'night', 'century', 'decade', 'microsecond', 'millisecond', 'nanosecond', 'millennium', 'fortnight', 'baseball', 'basketball', 'Batman', 'Joker', 'Superman', 'Magneto', 'Riddler', 'Spiderman', 'Wolverine', 'Deadpool', 'Lex_Luthor', 'Red_Skull', 'Sinestro', 'Green_Lantern', 'Dormammu', 'Venom', 'Doctor_Strange', 'Zatanna', 'Metallo', 'Wonder_Woman', 'Iron_Man', 'Ant_Man', 'Hulk', 'Daredevil', 'Thor', 'Ninja_Turle', 'Black_Widow', 'Ghost_Rider', 'CBS', 'ABC', 'NBC', 'CNN', 'MSNBC', 'NYT', 'WSJ', 'WashingtonPost', 'Fox', 'Bloomberg', 'PBS', 'USAToday', 'NPR', 'BBC', 'HuffingtonPost', 'Politico', 'New_Yorker', 'Slate', 'ice', 'snow', 'rain', 'wind', 'fire', 'sun', 'moon', 'star', 'syrup', 'glue', 'porridge', 'honey', 'jam', 'milk', 'juice', 'soda', 'coke', 'oil', 'tea', 'coffee', 'lemonade', 'blood', 'ice_cream', 'sundae', 'wine', 'alcohol', 'paint', 'ink', 'mayonnaise', 'sauce', 'salad_dressing', 'sour_cream', 'ketchup', 'barbecue_sauce', 'mustard', 'gravy', 'broth', 'smoothie', 'yogurt', 'air', 'frisbee', 'sky', 'Los_Angeles', 'Chicago', 'New_York', 'San_Fransisco', 'San_Jose', 'San_Diego', 'Madison', 'Cleveland', 'Honolulu', 'Indianapolis', 'Memphis', 'Pittsburgh', 'Louisville', 'Orlando', 'Oakland', 'Dallas', 'Fort_Worth', 'San_Antonio', 'Las_Vegas', 'Miami', 'Baltimore', 'New_Orleans', 'Houston', 'Atlanta', 'Austin', 'Boston', 'Denver', 'Philadelphia', 'Seattle', 'Detroit', 'Portland', 'Phoenix', 'Nashville', 'Milwaukee', 'Minneapolis', 'Charlotte', 'Cincinnati', 'OKC', 'Tampa', 'Jacksonville', 'sandwich', 'hamburger', 'sausage', 'bacon', 'chocolate', 'chowder', 'clams', 'cookies', 'cupcake', 'cereal', 'donut', 'ginger', 'gnocchi', 'granola', 'ham', 'noodles', 'pepperoni', 'pancake', 'spaghetti']
	properties = {"tall":"short", "expensive": "cheap", "dense" : "light", "mobile":"immobile", "heroic": "villainous", "dangerous":"safe", "round":"squarish", "liberal":"conservative", "shapeless":"shaped", "compressible": "incompressible", "dry" : "wet", "long":"brief", "delicious":"tasteless", "fury" : "furless", "loud" : "quiet", "sharp":"dull", "bright":"dark", "viscous":"watery", "social" : "solitary", "intelligent":"stupid", "hot":"cold", "rough":"smooth","aerodynamic":"clumsy", "healthy":"unhealthy", "thick":"thin", "northern":"southern","western":"eastern","big":"small","heavy":"light","strong":"breakable","rigid":"flexible","fast":"slow" }
	n = len(objects)
	m = len(properties)

	model = Model(embedding, embeddingSize)
	model = model.to(device)
	synthesis_accuracy = []
	training_indices = list(set(list(range(1, len(training_data)))) - set(warmup_indices))
	builtdata = training_data[warmup_indices]
	train(model, builtdata, 8, "false", "true")
	logger.info("Test Accuracy: %.2f" % evaluate(model, test_data, reverse_test_data, "true", "false", "false", "true"))
	for i in range(1, poolsize):
	  logger.info(i)
	  data = torch.LongTensor(m, 6).zero_()
	  count = 0
	  for pkey in properties:
	  	# set_trace()
	  	obj = random.sample(range(1, n), 2)
	  	if objects[obj[0]] in dict.word2idx and objects[obj[1]] in dict.word2idx:
		  	data[count, 0] = dict.get_idx(objects[obj[0]])
		  	data[count, 1] = dict.get_idx(objects[obj[1]])
		  	data[count, 2] = dict.get_idx(properties[pkey])
		  	data[count, 3] = dict.get_idx("similar")
		  	data[count, 4] = dict.get_idx(pkey)
		  	count = count + 1

	  uncertain_data = find_uncertain_synthesis_query(model, data, "false", "true")
	  print(objects[uncertain_data[0]]+"	"+objects[uncertain_data[1]])
	  uncertain_data[5] = int(input(">>>"))
	  torch.cat((builtdata, uncertain_data.unsqueeze(0)))
	  train(model, builtdata, 20, "false", "true")
	  acc = evaluate(model, test_data, reverse_test_data, "true", "false", "false", "true")
	  synthesis_accuracy.append(acc)
	  logger.info("Test Accuracy: %.2f" % acc)
	logger.info(synthesis_accuracy)
	return synthesis_accuracy

def plot(training_examples, random_accuracy, lc_accuracy, emc_accuracy):
	plt.plot(training_examples, random_accuracy, color='g', linewidth=1, label="Random")
	plt.plot(training_examples, lc_accuracy, color='red', linewidth=1, label="LC")
	plt.plot(training_examples, emc_accuracy, color='blue', linewidth=2, label="EMC")
	plt.legend()
	plt.xlabel("Training Examples")
	plt.ylabel("Accuracy")
	plt.tight_layout()
	plt.savefig('figs/ActiveLearning.png')

parser = argparse.ArgumentParser(description='ObjectPropertyCommonSense')
parser.add_argument('--data', type=str, default='data/')
parser.add_argument('--test', type=str, default= "test_data")
parser.add_argument('--train', type=str, default= "train_data")
parser.add_argument('--embtype', type=str, default="word2vec")
parser.add_argument('--embeddingSize', type=int, default=300)
args = parser.parse_args()
logger.debug(args)

location = args.data + args.embtype + "/" + args.embtype +'.6B.' + str(args.embeddingSize) + 'd'+ '-weights-norm' + '.refined.npy'
logger.debug("Embedding File: %s", location)
embedding = torch.FloatTensor(np.load(location))

# Loading Dictionary
dict = Dictionary(args.data, args.embtype)
corpus = DataLoader(dict, args.data, args.embtype, args.train, "false",args.train, args.test)

# Load Train and Test Data based on params
training_data = corpus.train
test_data = corpus.test
reverse_test_data = corpus.reverse_test
warmup_indices = random.sample(range(1, len(training_data)), 200)
poolsize = 1539

random_accuracy = al_random(embedding, args.embeddingSize, poolsize, warmup_indices, training_data, test_data, reverse_test_data)
lc_accuracy = al_lc(embedding, args.embeddingSize, poolsize, warmup_indices, training_data, test_data, reverse_test_data)
emc_accuracy = al_emc(embedding, args.embeddingSize, poolsize, warmup_indices, training_data, test_data, reverse_test_data)
synthesis_accuracy = al_synthesis_based(dict, embedding, args.embeddingSize, poolsize, warmup_indices, training_data, test_data, reverse_test_data)
plot(range(1, poolsize), random_accuracy, lc_accuracy, emc_accuracy)


