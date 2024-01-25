#!/bin/bash

# Create a new tmux session
session_name="record_bag_$(date +%s)"
tmux new-session -d -s $session_name
pass_word="classlab"

# Change the directory to ../topomaps/bags and run the rosbag record command
tmux select-pane -t 0
tmux send-keys "cd ../bags" Enter
tmux send-keys "rosbag record /camera/left/image_raw/compressed /camera/right/image_raw/compressed /scout_status /odom_chassis /gps/gps /imu/data_raw -o $1" Enter # change topic if necessary

# Attach to the tmux session
tmux -2 attach-session -t $session_name