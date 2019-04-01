# 1 * 6  = 3 Results
# Table 3
args="--four true --test test_data --train train_data --devtest false --test_relation all --zero false"
echo $args
for emb in " --embtype glove --embeddingSize 300" " --embtype lstm --embeddingSize 1024"  " --embtype word2vec --embeddingSize 300"
do
	echo $emb
	argswithRelation="$args$emb"
	for i in {1..10}
	do
		python script.py $argswithRelation
	done
done



