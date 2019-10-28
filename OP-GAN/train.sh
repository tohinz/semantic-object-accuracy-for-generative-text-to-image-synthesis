#!/bin/bash
GPU="$1"
echo "Starting training on GPU ${GPU}."
cd code
python main.py --cfg cfg/cfg_file_train.yml --gpu "$GPU"
echo "Done."
cd ..
