# Usage: ./scp_setup.sh [ec2 .pem file] [config .json file] [instance DNS]

# Build the script to be ran on the ec2 instance
bazel build code/py_alphago/play_games

sudo scp -i $1 -r bazel-bin setup.sh $2 ubuntu@$3:~/.
sudo ssh -L localhost:8888:localhost:8888 -i $1 ubuntu@$3

