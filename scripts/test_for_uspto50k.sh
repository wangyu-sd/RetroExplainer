#!/bin/bash

python entry.py\
  --batch_size 256 \
  --cuda 2 \
  --name 'predict' \
  --test \
  --dataset data/USPTO50K \
  --known_rxn_cnt \
  --known_rxn_type \
  --model_path model_saved/rxn_type_unknown_best.ckpt
