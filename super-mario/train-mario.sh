# Used to train a model on AWS

ENVIRONMENT=$1

echo Training $ENVIRONMENT model...
<<<<<<< HEAD
env WARNTIME=60 WARNSIG=1 KILLTIME=30 timelimit xvfb-run -s "-screen 0 1400x900x24" python main.py --env-name $ENVIRONMENT --record --start-fresh
=======
xvfb-run -s "-screen 0 1400x900x24" python main.py --env-name $ENVIRONMENT --record --start-fresh
>>>>>>> f795d89c54c7c6de091a14e7b4a5ba76580e2b3b

git add checkpoints/
git add playback/
git add save/
git commit -a -m "AUTO COMMIT UPDATING $ENVIRONMENT CHECKPOINTS AND RECORDS"
