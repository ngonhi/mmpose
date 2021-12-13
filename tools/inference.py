from mmpose.apis import (inference_bottom_up_pose_model, init_pose_model,
                         vis_pose_result)
from mmcv import Config
from mmpose.datasets import DatasetInfo
import os
import cv2
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description='mmpose inference')
    parser.add_argument(
        "--checkpoint_path",
        default="/mnt/ssd/marley/ID_Card/mmpose/work_dirs/baseline_2_higherhrnet/latest.pth",
        type=str,
        help="Path to model weight")
    parser.add_argument(
        "--config_path",
        default="/mnt/ssd/marley/ID_Card/mmpose/configs/ID_card/baseline_2_higherhrnet_w32_IDCard_512x512.py",
        type=str,
        help="Path to config file")
    parser.add_argument(
        "--image_dir",
        default="/mnt/ssd/marley/ID_Card/ID_card_data/test",
        type=str,
        help="Path to the directory containing images") 
    parser.add_argument(
        "--output_dir",
        default="/mnt/ssd/marley/ID_Card/ID_card_data/test_output",
        type=str,
        help="Path to the directory to save output images")
       
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    checkpoint = args.checkpoint_path
    config_file = args.config_path
    cfg = Config.fromfile(config_file)

    model = init_pose_model(cfg, checkpoint)
    dataset_info = model.cfg.data['test'].get('dataset_info', None)
    dataset_info = DatasetInfo(dataset_info)

    image_dir = args.image_dir
    images = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]
    for image in tqdm(images):
        try:
            keypoint_results, _ = inference_bottom_up_pose_model(model, image, 'BottomUpIDCardDataset', 
                                                                dataset_info, return_heatmap=False,
                                                                pose_nms_thr=0.5)
            # Filter confidence score less than 0.1
            keypoint_results = [kp for kp in keypoint_results if kp['score'] >= 0.1]
            img = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
            vis_result = vis_pose_result(model,
                             img,
                             keypoint_results,
                             radius=10,
                             thickness=5,
                             dataset=model.cfg.data.test.type,
                             show=False)
            vis_result = cv2.cvtColor(vis_result, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(output_dir, os.path.basename(image)), vis_result)
        except:
            print("Failed to process image: {}".format(image))

if __name__ == '__main__':
    main()