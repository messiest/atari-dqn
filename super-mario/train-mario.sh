# Used to train a model on AWS

ENVIRONMENT=$1

echo Training $ENVIRONMENT model...
env WARNTIME=60 WARNSIG=1 KILLTIME=30 timelimit xvfb-run -s "-screen 0 1400x900x24" python main.py --env-name $ENVIRONMENT --record True

git add checkpoints/
git add playback/
git add save/
git commit -a -m "AUTO COMMIT UPDATING $ENVIRONMENT CHECKPOINTS AND RECORDS"
