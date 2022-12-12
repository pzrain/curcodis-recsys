python src/main_node_new.py --data Ciao --epoch 120 --partitionK 4 --numGCNLayer 1 --numLayer 1 --ratio 1
echo "done 1 1 1"

python src/main_node_new.py --data Ciao --epoch 120 --partitionK 4 --numGCNLayer 5 --numLayer 7 --ratio 8
echo "done 5, 7, 8"

python src/main_node_new.py --data Ciao --epoch 120 --partitionK 4 --numGCNLayer 5 --numLayer 7 --ratio 4
echo "done 5, 7, 4"

python src/main_node_new.py --data Ciao --epoch 120 --partitionK 4 --numGCNLayer 5 --numLayer 7 --ratio 2
echo "done 5, 7, 2"

python src/main_node_new.py --data Ciao --epoch 120 --partitionK 4 --numGCNLayer 1 --numLayer 7 --ratio 8
echo "done 1, 7, 8"

python src/main_node_new.py --data Ciao --epoch 120 --partitionK 4 --numGCNLayer 1 --numLayer 7 --ratio 4
echo "done 1, 7, 4"

python src/main_node_new.py --data Ciao --epoch 120 --partitionK 4 --numGCNLayer 1 --numLayer 7 --ratio 2
echo "done 1, 7, 2"

python src/main_node_new.py --data Ciao --epoch 120 --partitionK 4 --numGCNLayer 5 --numLayer 1 --ratio 8
echo "done 5, 1, 8"

python src/main_node_new.py --data Ciao --epoch 120 --partitionK 4 --numGCNLayer 5 --numLayer 1 --ratio 4
echo "done 5, 1, 4"

python src/main_node_new.py --data Ciao --epoch 120 --partitionK 4 --numGCNLayer 5 --numLayer 1 --ratio 2
echo "done 5, 1, 2"
