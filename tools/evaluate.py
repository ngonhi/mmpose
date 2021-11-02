from mmcv import Config
from mmpose.datasets import build_dataloader, build_dataset
import json
import random
import numpy as np

config_file = '~/ID_Card/mmpose/configs/ID_card/overfit_higherhrnet_w32_IDCard_512x512.py'
cfg = Config.fromfile(config_file)

cfg.model.pretrained = None
cfg.data.test.test_mode = True

# build the dataloader
cfg.data.test.dataset_info.sigmas = [0.025] * 4
dataset = build_dataset(cfg.data.test, dict(test_mode=True))
dataloader_setting = dict(
    samples_per_gpu=16,
    workers_per_gpu=cfg.data.get('workers_per_gpu', 4),
    dist=False,
    shuffle=False,
    drop_last=False)
dataloader_setting = dict(dataloader_setting,
                              **cfg.data.get('test_dataloader', {}))
data_loader = build_dataloader(dataset, **dataloader_setting)

with open('/mnt/ssd/marley/ID_Card/mmpose/work_dirs/output/output_epoch4.json') as f:
    outputs = json.load(f)

# Random shift prediction
# for n, output in enumerate(outputs):
#     for i, pred in enumerate(output['preds']):
#         pred = np.array(pred)
#         pred[:, 0] += 50
#         pred[:, 1] += 50
#         output['preds'][i] = pred
#     outputs[n] = output

eval_config = cfg.get('evaluation', {})
work_dir = '/mnt/ssd/marley/ID_Card/mmpose/work_dirs/output'
results = dataset.evaluate(outputs[0:10], work_dir, **eval_config)
print(results)
results['AP'] = 0
print(results)
