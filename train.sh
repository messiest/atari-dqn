ENVIRONMENT=$1
EPISODES=$2

echo Training $ENVIRONMENT model...
python train.py $ENVIRONMENT --N $EPISODES

ls checkpoints/

git add checkpoints/
git commit -a -m "AUTO COMMIT UPDATING $ENVIRONMENT CHECKPOINTS"
