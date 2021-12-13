#!/usr/bin/env bash
# Copyright (c) OpenMMLab. All rights reserved.

CONFIG=/mnt/ssd/marley/ID_Card/mmpose/configs/ID_card/baseline_4_higherhrnet_w32_IDCard_640x640.py
CHECKPOINT=/mnt/ssd/marley/ID_Card/mmpose/work_dirs/baseline_4_higherhrnet_640x640/epoch_51.pth

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/test.py $CONFIG $CHECKPOINT #--out /mnt/ssd/marley/ID_Card/mmpose/work_dirs/baseline_2_higherhrnet/results.json
