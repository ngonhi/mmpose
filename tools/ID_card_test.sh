#!/usr/bin/env bash
# Copyright (c) OpenMMLab. All rights reserved.

CONFIG=/mnt/ssd/marley/ID_Card/mmpose/configs/ID_card/baseline_2_higherhrnet_w32_IDCard_512x512.py
CHECKPOINT=/mnt/ssd/marley/ID_Card/mmpose/work_dirs/baseline_2_higherhrnet_filter_kpt_v2_load_baseline1e13/epoch_1.pth

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/test.py $CONFIG $CHECKPOINT #--out /mnt/ssd/marley/ID_Card/mmpose/work_dirs/baseline_2_higherhrnet/results.json
