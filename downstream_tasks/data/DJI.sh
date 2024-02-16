#!/bin/bash
export TZ="GMT-8"

# Experiment variables
exp="DJ30_V2_RT"

python data_preparation.py \
--device            cuda:2 \
--exp               $exp \
--seed              42 \
--max_seq_len       30 \
--train_rate        0.8 \
--data_path         ../../data/DJI/DJI_50/tic/DJI_data_labeled_slice_and_merge_model_3dynamics_minlength50_quantile_labeling.csv \
--dynamic_dim       3 \
--label_dim         29 \
--history_length    30