#!/bin/bash

tmux new-session -d -s case5 "python train.py --case case5"
tmux new-session -d -s case6 "python train.py --case case6"
tmux new-session -d -s case10 "python train.py --case case10"
