# python src/main_edge.py --data lastfm  --epoch 120 --batchNum 50 > result_edge.txt
# python src/main_edge_cur_rearrange.py --data lastfm  --epoch 120 --batchNum 50 > result_rea.txt
# python src/main_edge_cur_babystep.py --data lastfm  --epoch 120 --batchNum 50 > result_baby.txt
DATASET=Ciao
BATCHSIZE=120

python src/edge.py --option edge --epoch 120 --data ${DATASET} --batchNum ${BATCHSIZE}

python src/edge.py --option rearrange --epoch 120 --data ${DATASET} --batchNum ${BATCHSIZE}

python src/edge.py --option babystep --epoch 120 --data ${DATASET} --batchNum ${BATCHSIZE}

DATASET=yelp

python src/edge.py --option edge --epoch 120 --data ${DATASET} --batchNum ${BATCHSIZE}

python src/edge.py --option rearrange --epoch 120 --data ${DATASET} --batchNum ${BATCHSIZE}

python src/edge.py --option babystep --epoch 120 --data ${DATASET} --batchNum ${BATCHSIZE}

python src/main_node.py --data lastfm --epoch 120 --batchNum 50
echo "done main_node lastfm"

python src/main_node.py --data Ciao --epoch 120 --batchNum 120
echo "done main_node Ciao"

python src/main_node.py --data yelp --epoch 120 --batchNum 120
echo "done main_node yelp"

python src/main_node.py --data Epinion --epoch 120 --batchNum 700
echo "done main_node Epinion"

DATASET=Epinion
BATCHSIZE=700
python src/edge.py --option edge --epoch 120 --data ${DATASET} --batchNum ${BATCHSIZE}
echo "done edge Epinion"

python src/edge.py --option rearrange --epoch 120 --data ${DATASET} --batchNum ${BATCHSIZE}
echo "done rearrange Epinion"

python src/edge.py --option babystep --epoch 120 --data ${DATASET} --batchNum ${BATCHSIZE}
echo "done babystep Epinion"