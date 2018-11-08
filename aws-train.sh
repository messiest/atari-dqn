ENVIRONMENT=$1
EPISODES=$2

echo Training $ENVIRONMENT model...
python epsilon-schedule.py
xvfb-run -s "-screen 0 1400x900x24" python train.py $ENVIRONMENT --N $EPISODES

ls checkpoints/

git add checkpoints/
git commit -a -m "AUTO COMMIT UPDATING $ENVIRONMENT CHECKPOINTS"
