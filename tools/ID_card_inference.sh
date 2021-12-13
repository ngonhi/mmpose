#!/usr/bin/env bash
# Copyright (c) OpenMMLab. All rights reserved.

CONFIG=/mnt/ssd/marley/ID_Card/mmpose/configs/ID_card/baseline_4_higherhrnet_w32_IDCard_640x640.py
CHECKPOINT=/mnt/ssd/marley/ID_Card/mmpose/work_dirs/baseline_4_higherhrnet_640x640/epoch_51.pth
IMAGE_DIR=/mnt/ssd/marley/ID_Card/ID_card_data/test
OUTPUT_DIR=/mnt/ssd/marley/ID_Card/ID_card_data/baseline_4_output

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/inference.py --checkpoint_path $CHECKPOINT --config_path $CONFIG --image_dir $IMAGE_DIR --output_dir $OUTPUT_DIR