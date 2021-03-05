#!/sur/bin/env bash

data_path=$1
gpus=$2
batch_size=$3

python main.py -root $data_path -gpus=$2 -batch_size=$batch_size
