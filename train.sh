ENVIRONMENT=$1
EPISODES=$2

echo Training $ENVIRONMENT model...
python train.py $ENVIRONMENT --N $EPISODES

ls checkpoints/
