#!/usr/bin/env bash
# Copyright (c) OpenMMLab. All rights reserved.

CONFIG=~/ID_Card/mmpose/configs/ID_card/overfit_higherhrnet_w32_IDCard_512x512.py

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -W ignore $(dirname "$0")/train.py $CONFIG
