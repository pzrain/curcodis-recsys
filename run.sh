# python src/main.py --data lastfm --epoch 120 --batchNum 50 > result/result_lastfm/result_base.txt
# echo "done main lastfm"

# python src/main.py --data Ciao --epoch 120 --batchNum 50 > result/result_ciao/result_base.txt
# echo "done main ciao"

python src/main_edge.py --data Ciao --epoch 120 --batchNum 120 > result/result_ciao/result_edge.txt
echo "done main_edge ciao"
python src/main_edge_cur_rearrange.py --data Ciao --epoch 120 --batchNum 120 > result/result_ciao/result_edge_cur_rea.txt
echo "done main_edge_cur_rea ciao"
python src/main_edge_cur_babystep.py --data Ciao --epoch 120 --batchNum 120 > result/result_ciao/result_edge_cur_babystep.txt
echo "done main_edge_cur_babystep ciao"

python src/main_edge.py --data lastfm --epoch 120 --batchNum 50 > result/result_lastfm/result_edge.txt
echo "done main_edge lastfm"
python src/main_edge_cur_rearrange.py --data lastfm --epoch 120 --batchNum 50 > result/result_lastfm/result_edge_cur_rea.txt
echo "done main_edge_cur_rea lastfm"
python src/main_edge_cur_babystep.py --data lastfm --epoch 120 --batchNum 50 > result/result_lastfm/result_edge_cur_babystep.txt
echo "done main_edge_cur_babystep lastfm"

# python src/main.py --data Epinion --epoch 120 --batchNum 80 > result/result_epinion/result_base.txt
# echo "done main epinion"
python src/main_edge.py --data Epinion --epoch 120 --batchNum 700 > result/result_epinion/result_edge.txt
echo "done main_edge epinion"
python src/main_edge_cur_rearrange.py --data Epinion --epoch 120 --batchNum 700 > result/result_epinion/result_edge_cur_rea.txt
echo "done main_edge_cur_rea epinion"
python src/main_edge_cur_babystep.py --data Epinion --epoch 120 --batchNum 700 > result/result_epinion/result_edge_cur_babystep.txt
echo "done main_edge_cur_babystep epinion"

# python src/main.py --data yelp --epoch 120 --batchNum 10 > result/result_yelp/result_base.txt
# echo "done main yelp"
python src/main_edge.py --data yelp --epoch 120 --batchNum 120 > result/result_yelp/result_edge.txt
echo "done main_edge yelp"
python -u src/main_edge_cur_rearrange.py --data yelp --epoch 120 --batchNum 120 > result/result_yelp/result_edge_cur_rea.txt
echo "done main_edge_cur_rea yelp"
python -u src/main_edge_cur_babystep.py --data yelp --epoch 120 --batchNum 120 > result/result_yelp/result_edge_cur_babystep.txt
echo "done main_edge_cur_babystep yelp"

python src/main_new.py --data Ciao --epoch 120 --batchNum 120 > result/result_ciao/result_new.txt
echo "done new Ciao"

python src/main_new.py --data lastfm --epoch 120 --batchNum 50 > result/result_lastfm/result_new.txt
echo "done new lastfm"

python src/main_new.py --data Epinion --epoch 120 --batchNum 70 > result/result_epinion/result_new.txt
echo "done new Epinion"

python src/main_new.py --data yelp --epoch 120 --batchNum 120 > result/result_yelp/result_new.txt
echo "done new yelp"