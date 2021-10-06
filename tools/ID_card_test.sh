#!/usr/bin/env bash
# Copyright (c) OpenMMLab. All rights reserved.

CONFIG=~/ID_Card/mmpose/configs/ID_card/overfit_higherhrnet_w32_IDCard_512x512.py
CHECKPOINT=~/ID_Card/mmpose/work_dirs/overfit_higherhrnet_w32_IDCard_512x512/best_AP_epoch_4.pth

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/test.py $CONFIG $CHECKPOINT --out /mnt/ssd/marley/ID_Card/mmpose/work_dirs/baseline_debug/results.json
