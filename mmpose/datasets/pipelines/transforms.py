import mmcv
import numpy as np
from numpy import random
import torch
import torch.nn.functional as F

# from mmdet.core import PolygonMasks
# from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from ..builder import PIPELINES


@PIPELINES.register_module()
class Resize:
    """Resize only input image
    """

    def __init__(self, is_train=True):
        self.is_train = is_train
    
    def __call__(self, results):
        """Call function to resize images
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Resized results
        """
        input_size = (int(results['ann_info']['image_size']), int(results['ann_info']['image_size']))
        img = results['img']
        img_shape = img.shape
        results['ann_info']['rescale'] = [img_shape[1]/input_size[0], img_shape[0]/input_size[1]]
        if not self.is_train:
            img, w, h = mmcv.imresize(img, input_size, interpolation='bicubic', backend='cv2', return_scale=True)
        else:
            if random.random() < 0.5: # Random interpolation with opencv and pillow
                backend = 'cv2' if random.random() < 0.5 else 'pillow'
                interpolation = ['nearest', 'bilinear', 'bicubic', 'area', 'lanczos'] if backend=='cv2' else \
                    ['nearest', 'bilinear', 'bicubic', 'lanczos']
                img = mmcv.imresize(img, input_size, interpolation=random.choice(interpolation), backend=backend)
            else: # Random interpolation with torch
                type_resize_torchs = ['nearest', 'bilinear', 'bicubic', 'area']
                type_resize_torch = random.choice(type_resize_torchs)
                align_corner = None
                if type_resize_torch in ['bicubic', 'bilinear']:
                    if random.random() < 0.5:
                        align_corner = True
            
                torch_img = torch.from_numpy(img)
                torch_img = torch_img.permute(2, 0, 1).float().unsqueeze(0)
                torch_img = F.interpolate(torch_img, size=input_size, 
                            mode=type_resize_torch, align_corners=align_corner)
                torch_img = torch_img.clamp(min=0, max=255)
            
                img = torch_img.numpy()
                img = img.squeeze()
                img = img.transpose(1, 2, 0).astype(np.uint8)

        results['img'] = img

        return results