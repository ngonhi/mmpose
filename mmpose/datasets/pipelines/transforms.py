# Copyright (c) OpenMMLab. All rights reserved.
import copy
import inspect
import math

import cv2
import mmcv
import numpy as np
from numpy import random

# from mmdet.core import PolygonMasks
# from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from ..builder import PIPELINES

try:
    from imagecorruptions import corrupt
except ImportError:
    corrupt = None

# try:
#     import albumentations
#     from albumentations import Compose
# except ImportError:
#     albumentations = None
#     Compose = None


@PIPELINES.register_module()
class Resize:
    """Resize only input image
    """

    def __init__(self):
        pass
    
    def __call__(self, results):
        """Call function to resize images
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Resized results
        """
        input_size = int(results['ann_info']['image_size'])
        img = results['img']

        img = cv2.resize(img, (input_size, input_size))
        results['img'] = img

        return results