#!/bin/bash
GPU="$1"
echo "Starting sampling on GPU ${GPU}."
cd code
python main.py --cfg cfg/cfg_file_eval.yml --gpu "$GPU"
echo "Done."
cd ..
