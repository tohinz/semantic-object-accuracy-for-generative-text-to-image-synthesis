#!/bin/bash
GPU="$1"
echo "Starting training on GPU ${GPU}."
python code/main.py --cfg code/cfg/cfg_file_train.yml --gpu "$GPU"
echo "Done."
