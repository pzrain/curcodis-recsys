python src/edge.py --data lastfm --option edge --epoch 120 --batchNum 60
echo "edge done!"
python src/edge.py --data lastfm --option rearrange --epoch 120 --batchNum 60
echo "rearrange done!"
python src/edge.py --data lastfm --option babystep --epoch 120 --batchNum 60
echo "babystep done!"