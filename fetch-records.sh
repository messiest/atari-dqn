ENV=$1
CHECKPOINTS=$2
PLAYBACK=$3

declare -a arr=(
    "ubuntu@ec2-34-238-138-255.compute-1.amazonaws.com"
    "ubuntu@ec2-54-208-171-85.compute-1.amazonaws.com"
    "ubuntu@ec2-18-234-138-56.compute-1.amazonaws.com"
    "ubuntu@ec2-54-174-155-141.compute-1.amazonaws.com"
    "ubuntu@ec2-184-73-31-39.compute-1.amazonaws.com"
    "ubuntu@ec2-54-85-248-80.compute-1.amazonaws.com"
    "ubuntu@ec2-54-85-177-185.compute-1.amazonaws.com"
    "ubuntu@ec2-35-175-226-207.compute-1.amazonaws.com"
)

for i in "${arr[@]}"
do
    mkdir -p /repos/mario.ai/logs/$ENV/
    scp -rp -i "/Users/mess/aws-keys/mess-creds.pem" $i:/home/ubuntu/mario.ai/logs/$ENV/ /repos/mario.ai/logs/$ENV/
    if [ "$CHECKPOINTS" = true ] ; then
        mkdir -p /repos/mario.ai/checkpoints/$ENV/
        scp -i "/Users/mess/aws-keys/mess-creds.pem" $i:/home/ubuntu/mario.ai/checkpoints/$ENV/*.tar /repos/mario.ai/checkpoints/$ENV/
    fi
    if [ "$PLAYBACK" = true ] ; then
        mkdir -p /repos/mario.ai/playback/$ENV/
        scp -rp -i "/Users/mess/aws-keys/mess-creds.pem" $i:/home/ubuntu/mario.ai/playback/$ENV/ /repos/mario.ai/playback/$ENV/
    fi
done
