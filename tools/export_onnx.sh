
CONFIG=/mnt/ssd/marley/ID_Card/mmpose/configs/ID_card/overfit_higherhrnet_w32_IDCard_512x512.py
CHECKPOINT=/mnt/ssd/marley/ID_Card/mmpose/work_dirs/baseline_higherhrnet/best_AP_epoch_13.pth
OUTPUT=/mnt/ssd/marley/ID_Card/lib_card_crop_alignment_keypoints/weights/best_AP_epoch13_cuda.onnx

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/deployment/pytorch2onnx.py $CONFIG $CHECKPOINT --output-file $OUTPUT --verify --shape 16 3 512 512
python ../remove_initializer_from_input.py --input $OUTPUT --output $OUTPUT