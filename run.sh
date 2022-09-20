# python src/main.py --data lastfm --epoch 120 --batchNum 50 > result/result_lastfm/result_base.txt
# echo "done main lastfm"
# python src/main_edge.py --data lastfm --epoch 120 --batchNum 50 > result/result_lastfm/result_edge.txt
# echo "done main_edge lastfm"
# python src/main_curriculum.py --data lastfm --epoch 120 --batchNum 50 > result/result_lastfm/result_cur.txt
# echo "done main_curriculum lastfm"
# python src/main_edge_cur.py --data lastfm --epoch 120 --batchNum 50 > result/result_lastfm/result_edge_cur.txt
# echo "done main_edge_cur lastfm"
# python src/main_edge_cur_rearrange.py --data lastfm --epoch 120 --batchNum 50 > result/result_lastfm/result_edge_cur_rea.txt
# echo "done main_edge_cur_rea lastfm"
# python src/main_edge_cur_babystep.py --data lastfm --epoch 120 --batchNum 50 > result/result_lastfm/result_edge_cur_babystep.txt
# echo "done main_edge_cur_babystep lastfm"

# python src/main.py --data Ciao --epoch 120 --batchNum 50 > result/result_ciao/result_base.txt
# echo "done main ciao"
# python src/main_edge.py --data Ciao --epoch 120 --batchNum 80 > result/result_ciao/result_edge.txt
# echo "done main_edge ciao"
# python src/main_curriculum.py --data Ciao --epoch 120 --batchNum 50 > result/result_ciao/result_cur.txt
# echo "done main_curriculum ciao"
# python src/main_edge_cur.py --data Ciao --epoch 120 --batchNum 80 > result/result_ciao/result_edge_cur.txt
# echo "done main_edge_cur ciao"
# python src/main_edge_cur_rearrange.py --data Ciao --epoch 120 --batchNum 80 > result/result_ciao/result_edge_cur_rea.txt
# echo "done main_edge_cur_rea ciao"
# python src/main_edge_cur_babystep.py --data Ciao --epoch 120 --batchNum 80 > result/result_ciao/result_edge_cur_babystep.txt
# echo "done main_edge_cur_babystep ciao"

# python src/main.py --data Epinion --epoch 120 --batchNum 80 > result/result_epinion/result_base.txt
# echo "done main epinion"
# python src/main_edge.py --data Epinion --epoch 120 --batchNum 600 > result/result_epinion/result_edge.txt
# echo "done main_edge epinion"
# python src/main_curriculum.py --data Epinion --epoch 120 --batchNum 80 > result/result_epinion/result_cur.txt
# echo "done main_curriculum epinion"
# python src/main_edge_cur.py --data Epinion --epoch 120 --batchNum 600 > result/result_epinion/result_edge_cur.txt
# echo "done main_edge_cur epinion"
# python src/main_edge_cur_rearrange.py --data Epinion --epoch 120 --batchNum 600 > result/result_epinion/result_edge_cur_rea.txt
# echo "done main_edge_cur_rea epinion"
# python src/main_edge_cur_babystep.py --data Epinion --epoch 120 --batchNum 600 > result/result_epinion/result_edge_cur_babystep.txt
# echo "done main_edge_cur_babystep epinion"

python src/main.py --data yelp --epoch 120 --batchNum 10 > result/result_yelp/result_base.txt
echo "done main yelp"
python src/main_edge.py --data yelp --epoch 120 --batchNum 120 > result/result_yelp/result_edge.txt
echo "done main_edge yelp"
python -u src/main_curriculum.py --data yelp --epoch 120 --batchNum 10 > result/result_yelp/result_cur.txt
echo "done main_curriculum yelp"
python -u src/main_edge_cur.py --data yelp --epoch 120 --batchNum 120 > result/result_yelp/result_edge_cur.txt
echo "done main_edge_cur yelp"
python -u src/main_edge_cur_rearrange.py --data yelp --epoch 120 --batchNum 120 > result/result_yelp/result_edge_cur_rea.txt
echo "done main_edge_cur_rea yelp"
python -u src/main_edge_cur_babystep.py --data yelp --epoch 120 --batchNum 120 > result/result_yelp/result_edge_cur_babystep.txt
echo "done main_edge_cur_babystep yelp"
# 37482