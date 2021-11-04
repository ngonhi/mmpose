#!/usr/bin/env bash
# Copyright (c) OpenMMLab. All rights reserved.

CONFIG=/mnt/ssd/marley/ID_Card/mmpose/configs/ID_card/baseline_2_higherhrnet_w32_IDCard_512x512.py
CHECKPOINT=/mnt/ssd/marley/ID_Card/mmpose/work_dirs/baseline_2_higherhrnet_filter_kpt_v2/epoch_1.pth
IMAGE_DIR=/mnt/ssd/marley/ID_Card/ID_card_data/filter_kpt/test
OUTPUT_DIR=/mnt/ssd/marley/ID_Card/ID_card_data/filter_kpt/baseline_2_higherhrnet_filter_kpt_v2_test_output

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/inference.py --checkpoint_path $CHECKPOINT --config_path $CONFIG --image_dir $IMAGE_DIR --output_dir $OUTPUT_DIR