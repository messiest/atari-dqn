FILE=$1

mkdir frames
ffmpeg -i $FILE.mp4  -r 15 'frames/frame-%03d.jpg'
cd frames
convert -delay 1 -loop 0 *.jpg ../$FILE.gif
cd ../
rm -rf frames
echo File $FILE.gif created.
