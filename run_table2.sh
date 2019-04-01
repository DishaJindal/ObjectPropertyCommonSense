# 1* 10 = 10 Results(Table2 4th row)
# PCE zero-shot
args2="--data data/ --embtype lstm --embeddingSize 1024 --test verb_physics_test_5 --reverse true --train verb_physics_train_5 --dev verb_physics_dev_5 --zero true"

for rel in " --test_relation big"  " --test_relation heavy"  " --test_relation rigid"  " --test_relation strong" " --test_relation fast"
do
	echo $rel
	argswithRelation="$args2$rel"
	for i in {1..10}
	do
		python script.py $argswithRelation
	done
done

# 1* 10 = 10 Results(Table2 3rd row)
# PCE zero-shot
args3="--data data/ --embtype lstm --embeddingSize 1024 --test verb_physics_test_5 --onepole true --reverse true --train verb_physics_train_5 --zero true"

for rel in " --test_relation big"  " --test_relation heavy"  " --test_relation rigid"  " --test_relation strong" " --test_relation fast"
do
	echo $rel
	argswithRelation="$args3$rel"
	for i in {1..10}
	do
		python script.py $argswithRelation
	done
done

1* 10 = 10 Results(Table2 2th row)
PCE zero-shot
args2="--data data/ --embtype lstm --embeddingSize 1024 --test verb_physics_test_5 --reverse true --train verb_physics_train_5 --dev verb_physics_dev_5 --zero true"

for rel in " --test_relation big"  " --test_relation heavy"  " --test_relation rigid"  " --test_relation strong" " --test_relation fast"
do
	echo $rel
	argswithRelation="$args2$rel"
	for i in {1..10}
	do
		python baseline_script.py $argswithRelation
	done
done