## Description
This project reproduces the results from the paper [Extracting Commonsense Properties from Embeddings with Limited
Human Guidance](https://www.aclweb.org/anthology/P18-2102)
## Folders

**data**: Contains data files

**figs**: Contains output plots, also added in the report

**output_dumps**: The results of all runs for reference

## Code Walkthrough

### Scripts
#### Main Script: script.py

	Argument Description
	--data data/ : The folder containng the data files
	--test verb_physics_test_5 : Test Data File
	--devtest : {"true", "false"}, Whether to evaluate on dev data or not
	--dev : Dev Data File
	--train verb_physics_train_5 : Training Data File
	--embtype lstm : {word2vec, glove, lstm} 
	--embeddingSize 1024 : Embedding size; give 1024 for lstm and 300 for other two
	--test_relation all : {big, heavy, rigid, strong, fast, all} These are the options for verb physics
	--zero False : {"true", "false"}, Whether you want zero shot learning on the given test_relation
	--reverse true : {"true", "false"}, Whether you want to consider reverse tuples while evaluation
	--poleSensitivity False : {"true", "false"}: If you want to test on poleWord1 and poleWord2.
	--poleWord1 slow : Relation for R<
	--poleWord2 speedy : Relation for R>
	--onepole : {"true", "false"}: To use PCE(one-pole)
	--four : {"true", "false"}: To use 4 way model

	# To Run the script with default configuration
	python script.py

	# To Run one configuration of 3 way model(Example Configuration)
	python script.py --train verb_physics_train_5 --dev verb_physics_dev_5 --test verb_physics_test_5 --devtest true --four false --reverse true --onepole false

	# To Run one configuration of 4 way model(Example Configuration)
	python script.py --train train_data --test test_data --devtest false --four true --reverse true --onepole false

#### Other Scripts:
datautil.py: Contains functions for creating dictionary, preprocessing and filtering data 

models.py: Contains Emb_Similarity and Model(the main one) 

train_evaluate.py: Train, Evaluate and active learning utilities(find_LC_query, find_EMC_query, find_uncertain_synthesis_query)

### Embedding Similarity Baseline
python emb_similarity.py

### Majority Baseline
python majority.py

### Active Learning Experiments
Runs 4 active approaches(Random, LC, EMC and Synthesis Based) and Plots the training examples vs Accuracy Plot. For details, please refer the [report](https://github.com/DishaJindal/ObjectPropertyCommonSense/blob/master/Report.pdf).
python active_learning.py

### Score Analysis
Trains the model on verb physics dataset usign same parameters and calculates a proxy score of relatedness and plots the relative size and weights of some objects
python score_analysis.py

### Shell script that runs all configurations in Table1 10 times
sh run_table1.sh > table1_result
sh run_table2.sh > table2_result
sh run_table3.sh > table3_result

### Shell scripts for statistics and plots
Python script uses the results from the table1_result, table1_result and table3_result files, calculates mean, std and plots and save figures in figs folder
python stats_table1.py
python stats_table2.py
python stats_table3.py

### Significance Test File
python mcnemar.py
