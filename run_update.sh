# python src/edge.py --option edge --data Ciao --epoch 120 --batchNum 120 --split 0
# echo "done edge Ciao"

# python src/edge.py --option rearrange --data Ciao --epoch 120 --batchNum 120 --split 0
# echo "done rearrange Ciao"

# python src/edge.py --option babystep --data Ciao --epoch 120 --batchNum 120 --split 0
# echo "done babystep Ciao"

# python src/main_edge.py --data yelp --epoch 120 --batchNum 120
# echo "done edge yelp"

# python src/main_edge_cur_rearrange.py --data yelp --epoch 120 --batchNum 120
# echo "done rearrange yelp"

# python src/main_edge_cur_babystep.py --data yelp --epoch 120 --batchNum 120
# echo "done babystep yelp"

python src/main_edge.py --data Ciao --epoch 120 --batchNum 120 --split 0
echo "done edge Ciao"

python src/main_edge_cur_rearrange.py --data Ciao --epoch 120 --batchNum 120 --split 0
echo "done rearrange Ciao"

python src/main_edge_cur_babystep.py --data Ciao --epoch 120 --batchNum 120 --split 0
echo "done babystep Ciao"