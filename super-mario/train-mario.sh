ENVIRONMENT=$1

# echo Building Super Mario Bros. environments...
# xvfb-run -s "-screen 0 1400x900x24" python build_levels.py

echo Training $ENVIRONMENT model...
xvfb-run -s "-screen 0 1400x900x24" python main.py --env-name $ENVIRONMENT --record True

ls checkpoints/

git add checkpoints/
git add playback/
git add save/
git commit -a -m "AUTO COMMIT UPDATING $ENVIRONMENT CHECKPOINTS AND RECORDS"
