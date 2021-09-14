# Copyright (c) OpenMMLab. All rights reserved.
from .bottom_up_aic import BottomUpAicDataset
from .bottom_up_coco import BottomUpCocoDataset
from .bottom_up_coco_wholebody import BottomUpCocoWholeBodyDataset
from .bottom_up_crowdpose import BottomUpCrowdPoseDataset
from .bottom_up_mhp import BottomUpMhpDataset
from .bottom_up_ID_card import BottomUpIDCardDataset

__all__ = [
    'BottomUpCocoDataset', 'BottomUpCrowdPoseDataset', 'BottomUpMhpDataset',
    'BottomUpAicDataset', 'BottomUpCocoWholeBodyDataset', 'BottomUpIDCardDataset'
]
