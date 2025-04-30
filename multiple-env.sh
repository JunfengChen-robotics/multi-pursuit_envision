#!/bin/bash

python train.py --case case5 --render --phase0_steps 500000 --phase1_steps 1000000 --phase2_steps 5000000
python train.py --case case6 --render
python train.py --case case10 --render --phase0_steps 5000000 --phase1_steps 10000000 --phase2_steps 40000000