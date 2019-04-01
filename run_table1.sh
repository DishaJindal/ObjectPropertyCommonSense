# 3 * 12 = 36 Results
# PCE(LSTM), PCE(Glove), PCE(Word2vec)
args="--data data/ --test verb_physics_test_5 --reverse true --train verb_physics_train_5 --zero False --majority False --poleSensitivity False --poleWord1 slow --poleWord2 speedy"

for rel in " --test_relation big"  " --test_relation heavy"  " --test_relation rigid"  " --test_relation strong" " --test_relation fast" " --test_relation all"
do
	for emb in " --embtype lstm --embeddingSize 1024" " --embtype glove --embeddingSize 300" " --embtype word2vec --embeddingSize 300"
	do
		echo $rel
		echo $emb
		argswithRelation="$args$rel$emb"
		for i in {1..10}
		do
			python script.py $argswithRelation
		done
	done
done

# 1 * 6  = 6 Results
# PCE(no reverse)
args1="--data data/ --embtype lstm --embeddingSize 1024 --test verb_physics_test_5 --reverse false --train verb_physics_train_5 --zero False"
echo $args1
for rel in " --test_relation big"  " --test_relation heavy"  " --test_relation rigid"  " --test_relation strong" " --test_relation fast" " --test_relation all"
do
	echo $rel
	argswithRelation="$args1$rel"
	for i in {1..10}
	do
		python script.py $argswithRelation
	done
done

#  1 * 6 = 6 Results
# PCE(one pole)
args1="--data data/ --embtype lstm --embeddingSize 1024 --test verb_physics_test_5 --onepole true --reverse true --train verb_physics_train_5 --zero False"
echo $args1
for rel in " --test_relation big"  " --test_relation heavy"  " --test_relation rigid"  " --test_relation strong" " --test_relation fast" " --test_relation all"
do
	echo $rel
	argswithRelation="$args1$rel"
	for i in {1..10}
	do
		python script.py $argswithRelation
	done
done

# Majority Baseline
for rel in " --test_relation big"  " --test_relation heavy"  " --test_relation rigid"  " --test_relation strong" " --test_relation fast" " --test_relation all"
do
	# for i in {1..10}
	# do
	python majority.py $rel
	# done
done


