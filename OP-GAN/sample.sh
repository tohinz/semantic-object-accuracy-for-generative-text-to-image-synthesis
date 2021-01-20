#!/bin/bash
GPU="$1"
echo "Starting sampling on GPU ${GPU}."
python code/main.py --cfg code/cfg/cfg_file_eval.yml --gpu "$GPU"
echo "Done."
