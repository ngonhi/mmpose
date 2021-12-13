#!/usr/bin/env bash
# Copyright (c) OpenMMLab. All rights reserved.

CONFIG=/mnt/ssd/marley/ID_Card/mmpose/configs/ID_card/baseline_4_higherhrnet_w32_IDCard_640x640.py

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -W ignore $(dirname "$0")/train.py $CONFIG
