#!/bin/bash

python entry.py\
  --batch_size 1024 \
  --cuda 0 \
  --name 'predict' \
  --test \
  --dataset data/USPTO50K \
  --model_path /mnt/solid/wy/retro2/tb_logs/retro1/version_481/checkpoints/epoch=509-step=39270.ckpt
