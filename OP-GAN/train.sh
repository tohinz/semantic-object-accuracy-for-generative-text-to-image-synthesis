#!/bin/bash
GPU=$1
export CUDA_VISIBLE_DEVICES=${GPU}
export PYTHONUNBUFFERED=1
if [ -z "$GPU" ]
then
      echo "Starting training on CPU."
else
      echo "Starting training on GPU ${GPU}."
fi
python3 -u code/main.py --cfg code/cfg/cfg_file_train.yml
echo "Done."
