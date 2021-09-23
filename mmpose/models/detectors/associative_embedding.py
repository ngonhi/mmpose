# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import numpy as np

import mmcv
import torch
from mmcv.image import imwrite
from mmcv.utils.misc import deprecated_api_warning
from mmcv.visualization.image import imshow

from mmpose.core.evaluation import (aggregate_results, get_group_preds,
                                    get_multi_stage_outputs)
from mmpose.core.post_processing import oks_nms
from mmpose.core.post_processing.group import HeatmapParser
from mmpose.core.visualization import imshow_keypoints
from .. import builder
from ..builder import POSENETS
from .base import BasePose

from torchvision import transforms
invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])

try:
    from mmcv.runner import auto_fp16
except ImportError:
    warnings.warn('auto_fp16 from mmpose will be deprecated from v0.15.0'
                  'Please install mmcv>=1.1.4')
    from mmpose.core import auto_fp16


@POSENETS.register_module()
class AssociativeEmbedding(BasePose):
    """Associative embedding pose detectors.

    Args:
        backbone (dict): Backbone modules to extract feature.
        keypoint_head (dict): Keypoint head to process feature.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        pretrained (str): Path to the pretrained models.
        loss_pose (None): Deprecated arguments. Please use
            `loss_keypoint` for heads instead.
    """

    def __init__(self,
                 backbone,
                 keypoint_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 loss_pose=None):
        super().__init__()
        self.fp16_enabled = False

        self.backbone = builder.build_backbone(backbone)

        if keypoint_head is not None:

            if 'loss_keypoint' not in keypoint_head and loss_pose is not None:
                warnings.warn(
                    '`loss_pose` for BottomUp is deprecated, '
                    'use `loss_keypoint` for heads instead. See '
                    'https://github.com/open-mmlab/mmpose/pull/382'
                    ' for more information.', DeprecationWarning)
                keypoint_head['loss_keypoint'] = loss_pose

            self.keypoint_head = builder.build_head(keypoint_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.use_udp = test_cfg.get('use_udp', False)
        self.parser = HeatmapParser(self.test_cfg)
        self.init_weights(pretrained=pretrained)

    @property
    def with_keypoint(self):
        """Check if has keypoint_head."""
        return hasattr(self, 'keypoint_head')

    def init_weights(self, pretrained=None):
        """Weight initialization for model."""
        self.backbone.init_weights(pretrained)
        if self.with_keypoint:
            self.keypoint_head.init_weights()

    @auto_fp16(apply_to=('img', ))
    def forward(self,
                img=None,
                targets=None,
                masks=None,
                joints=None,
                img_metas=None,
                return_loss=True,
                return_heatmap=False,
                **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss is True.
        Note:
            batch_size: N
            num_keypoints: K
            num_img_channel: C
            img_width: imgW
            img_height: imgH
            heatmaps weight: W
            heatmaps height: H
            max_num_people: M
        Args:
            img(torch.Tensor[NxCximgHximgW]): Input image.
            targets(List(torch.Tensor[NxKxHxW])): Multi-scale target heatmaps.
            masks(List(torch.Tensor[NxHxW])): Masks of multi-scale target
                                              heatmaps
            joints(List(torch.Tensor[NxMxKx2])): Joints of multi-scale target
                                                 heatmaps for ae loss
            img_metas(dict):Information about val&test
                By default this includes:
                - "image_file": image path
                - "aug_data": input
                - "test_scale_factor": test scale factor
                - "base_size": base size of input
                - "center": center of image
                - "scale": scale of image
                - "flip_index": flip index of keypoints

            return loss(bool): Option to 'return_loss'. 'return_loss=True' for
                training, 'return_loss=False' for validation & test
            return_heatmap (bool) : Option to return heatmap.

        Returns:
            dict|tuple: if 'return_loss' is true, then return losses.
              Otherwise, return predicted poses, scores, image
              paths and heatmaps.
        """
        if return_loss:
            return self.forward_train(img, targets, masks, joints, img_metas, return_heatmap=return_heatmap,
                                      **kwargs)
        return self.forward_test(
            img, img_metas, return_heatmap=return_heatmap, **kwargs)
    
    def _visualize(self, pose_results, img):
        skeleton = [[0, 1], [1, 2], [2, 3], [3, 0]]
        pose_link_color = [[128, 255, 0], [0, 255, 128], [255, 128, 0], [255, 128, 128]]
        pose_kpt_color = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [128, 128, 128]]

        img = self.show_result(
            img,
            pose_results,
            skeleton=skeleton,
            pose_link_color=pose_link_color,
            pose_kpt_color=pose_kpt_color,
            radius=4,
            thickness=1,
            kpt_score_thr=0.3
        )

        return img

    def _inference(self, results, images, img_metas):
        img_list = []
        heatmap_list = []
        for i, result in enumerate(results):
            pose_results = []
            img = invTrans(images[i]).cpu().detach().numpy()
            img = np.transpose(img, (2,1,0))
            if len(result['preds']) != 0:
                for idx, pred in enumerate(result['preds']):
                    area = (np.max(pred[:, 0]) - np.min(pred[:, 0])) * (
                        np.max(pred[:, 1]) - np.min(pred[:, 1]))
                    pose_results.append({
                        'keypoints': pred[:, :3],
                        'score': result['scores'][idx],
                        'area': area,
                    })
                # pose nms
                keep = oks_nms(pose_results, thr=0.9, sigmas=self.train_cfg.sigmas)
                pose_results = [pose_results[_keep] for _keep in keep]
                img = self._visualize(pose_results, img)
            heatmap_list.append(result['output_heatmap'])
            img_list.append(img)
        
        img_list = np.array(img_list)
        heatmap_list = np.array(heatmap_list)
        return img_list, heatmap_list

    def _get_results(self, outputs, img_metas):
        scale = img_metas[0]['test_scale_factor'][0]
        test_scale_factor = img_metas[0]['test_scale_factor']
        aggregated_heatmaps = None
        tags_list = []

        _, heatmaps, tags = get_multi_stage_outputs(
            outputs=outputs,
            num_joints=self.test_cfg['num_joints'],
            with_heatmaps=self.test_cfg['with_heatmaps'],
            with_ae=self.test_cfg['with_ae'],
            tag_per_joint=self.test_cfg['tag_per_joint'],    
            project2image=self.test_cfg['project2image'],
            size_projected=img_metas[0]['base_size'],
            outputs_flip=None,
            flip_index=None,
            align_corners=self.use_udp)

        aggregated_heatmaps, tags_list = aggregate_results(
            scale=scale,
            aggregated_heatmaps=aggregated_heatmaps,
            tags_list=tags_list,
            heatmaps=heatmaps,
            tags=tags,
            test_scale_factor=test_scale_factor,
            project2image=self.test_cfg['project2image'],
            flip_test=False,
            align_corners=self.use_udp)
        tags = torch.cat(tags_list, dim=4)

        results = []
        for i in range(len(img_metas)):
            result = {}
            
            _aggregated_heatmaps = torch.unsqueeze(aggregated_heatmaps[i], dim=0)
            _tags = torch.unsqueeze(tags[i], 0)
            # perform grouping
            grouped, scores = self.parser.parse(_aggregated_heatmaps.detach(),
                                                _tags.detach(),
                                                self.test_cfg['adjust'],
                                                self.test_cfg['refine'])

            preds = get_group_preds(
                grouped,
                img_metas[i]['center'],
                img_metas[i]['scale'], [_aggregated_heatmaps.size(3),
                        _aggregated_heatmaps.size(2)],
                use_udp=self.use_udp)
            image_paths = []
            image_paths.append(img_metas[i]['image_file'])

            output_heatmap = _aggregated_heatmaps.detach().cpu().numpy()

            result['preds'] = preds
            result['scores'] = scores
            result['image_paths'] = image_paths
            result['output_heatmap'] = output_heatmap
            results.append(result)
        
        return results

    def forward_train(self, img, targets, masks, joints, img_metas, return_heatmap, **kwargs):
        """Forward the bottom-up model and calculate the loss.

        Note:
            batch_size: N
            num_keypoints: K
            num_img_channel: C
            img_width: imgW
            img_height: imgH
            heatmaps weight: W
            heatmaps height: H
            max_num_people: M

        Args:
            img(torch.Tensor[NxCximgHximgW]): Input image.
            targets(List(torch.Tensor[NxKxHxW])): Multi-scale target heatmaps.
            masks(List(torch.Tensor[NxHxW])): Masks of multi-scale target
                                              heatmaps
            joints(List(torch.Tensor[NxMxKx2])): Joints of multi-scale target
                                                 heatmaps for ae loss
            img_metas(dict):Information about val&test
                By default this includes:
                - "image_file": image path
                - "aug_data": input
                - "test_scale_factor": test scale factor
                - "base_size": base size of input
                - "center": center of image
                - "scale": scale of image
                - "flip_index": flip index of keypoints

        Returns:
            dict: The total loss for bottom-up
        """
        output = self.backbone(img)
        if self.with_keypoint:
            output = self.keypoint_head(output)

        # if return loss
        losses = dict()
        if self.with_keypoint:
            keypoint_losses, sum_loss = self.keypoint_head.get_loss(
                output, targets, masks, joints)
            losses.update(keypoint_losses)
    
        # Calculate metrics
    
        # Get top k loss
        batch_size = img.size(0)
        k = self.train_cfg['topk'] if self.train_cfg['topk'] <= batch_size else batch_size
        topk_loss = torch.topk(sum_loss, k, sorted=True)
        topk_loss_value = topk_loss.values.tolist()
        topk_loss_index = topk_loss.indices.tolist()
        
        topk_img = [img[i] for i in topk_loss_index]
        topk_img_metas = [img_metas[i] for i in topk_loss_index]
        topk_results = self._get_results(output, topk_img_metas)
        topk_img, topk_heatmap = self._inference(topk_results, topk_img, topk_img_metas)

        return losses, topk_loss_value, topk_img, topk_heatmap

    def forward_dummy(self, img):
        """Used for computing network FLOPs.

        See ``tools/get_flops.py``.

        Args:
            img (torch.Tensor): Input image.

        Returns:
            Tensor: Outputs.
        """
        output = self.backbone(img)
        if self.with_keypoint:
            output = self.keypoint_head(output)
        return output

    def forward_test(self, img, img_metas, return_heatmap=False, **kwargs):
        """Inference the bottom-up model.

        Note:
            Batchsize = N (currently support batchsize = 1)
            num_img_channel: C
            img_width: imgW
            img_height: imgH

        Args:
            flip_index (List(int)):
            aug_data (List(Tensor[NxCximgHximgW])): Multi-scale image
            test_scale_factor (List(float)): Multi-scale factor
            base_size (Tuple(int)): Base size of image when scale is 1
            center (np.ndarray): center of image
            scale (np.ndarray): the scale of image
        """
        assert img.size(0) == len(img_metas)
        test_scale_factor = img_metas[0]['test_scale_factor']
        base_size = img_metas[0]['base_size']
        center = img_metas[0]['center']
        scale = img_metas[0]['scale']
        flip_index = img_metas[0]['flip_index']

        aug_h = img_metas[0]['aug_data'][0].size(3)
        aug_w = img_metas[0]['aug_data'][0].size(2)
        channel = img_metas[0]['aug_data'][0].size(1)
        aug_data = []
        for i in range(len(test_scale_factor)):
            aug_data.append(torch.empty((img.size(0),channel,aug_w, aug_h)))
            for j in range(len(img_metas)):
                aug_data[i][j] = img_metas[j]['aug_data'][i]
        aggregated_heatmaps = None
        tags_list = []
        for idx, s in enumerate(sorted(test_scale_factor, reverse=True)):
            image_resized = aug_data[idx].to(img.device)
            features = self.backbone(image_resized)
            if self.with_keypoint:
                outputs = self.keypoint_head(features)
            if self.test_cfg.get('flip_test', True):
                # use flip test
                features_flipped = self.backbone(
                    torch.flip(image_resized, [3]))
                if self.with_keypoint:
                    outputs_flipped = self.keypoint_head(features_flipped)
            else:
                outputs_flipped = None

            _, heatmaps, tags = get_multi_stage_outputs(
                outputs,
                outputs_flipped,
                self.test_cfg['num_joints'],
                self.test_cfg['with_heatmaps'],
                self.test_cfg['with_ae'],
                self.test_cfg['tag_per_joint'],
                flip_index,
                self.test_cfg['project2image'],
                base_size,
                align_corners=self.use_udp)

            aggregated_heatmaps, tags_list = aggregate_results(
                s,
                aggregated_heatmaps,
                tags_list,
                heatmaps,
                tags,
                test_scale_factor,
                self.test_cfg['project2image'],
                self.test_cfg.get('flip_test', True),
                align_corners=self.use_udp)
        
        # average heatmaps of different scales
        aggregated_heatmaps = aggregated_heatmaps / float(
            len(test_scale_factor))
        tags = torch.cat(tags_list, dim=4)

        results = []
        for i in range(len(img_metas)):
            result = {}
            
            _aggregated_heatmaps = torch.unsqueeze(aggregated_heatmaps[i], dim=0)
            _tags = torch.unsqueeze(tags[i], 0)
            # perform grouping
            grouped, scores = self.parser.parse(_aggregated_heatmaps, _tags,
                                                self.test_cfg['adjust'],
                                                self.test_cfg['refine'])

            preds = get_group_preds(
                grouped,
                center,
                scale, [_aggregated_heatmaps.size(3),
                        _aggregated_heatmaps.size(2)],
                use_udp=self.use_udp)
            image_paths = []
            image_paths.append(img_metas[i]['image_file'])

            if return_heatmap:
                output_heatmap = _aggregated_heatmaps.detach().cpu().numpy()
            else:
                output_heatmap = None

            result['preds'] = preds
            result['scores'] = scores
            result['image_paths'] = image_paths
            result['output_heatmap'] = output_heatmap
            results.append(result)

        return results

    @deprecated_api_warning({'pose_limb_color': 'pose_link_color'},
                            cls_name='AssociativeEmbedding')
    def show_result(self,
                    img,
                    result,
                    skeleton=None,
                    kpt_score_thr=0.3,
                    bbox_color=None,
                    pose_kpt_color=None,
                    pose_link_color=None,
                    radius=4,
                    thickness=1,
                    font_scale=0.5,
                    win_name='',
                    show=False,
                    show_keypoint_weight=False,
                    wait_time=0,
                    out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (list[dict]): The results to draw over `img`
                (bbox_result, pose_result).
            skeleton (list[list]): The connection of keypoints.
                skeleton is 0-based indexing.
            kpt_score_thr (float, optional): Minimum score of keypoints
                to be shown. Default: 0.3.
            pose_kpt_color (np.array[Nx3]`): Color of N keypoints.
                If None, do not draw keypoints.
            pose_link_color (np.array[Mx3]): Color of M links.
                If None, do not draw links.
            radius (int): Radius of circles.
            thickness (int): Thickness of lines.
            font_scale (float): Font scales of texts.
            win_name (str): The window name.
            show (bool): Whether to show the image. Default: False.
            show_keypoint_weight (bool): Whether to change the transparency
                using the predicted confidence scores of keypoints.
            wait_time (int): Value of waitKey param.
                Default: 0.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            Tensor: Visualized image only if not `show` or `out_file`
        """
        img = mmcv.imread(img)
        img = img.copy()
        img_h, img_w, _ = img.shape

        pose_result = []
        for res in result:
            pose_result.append(res['keypoints'])

        imshow_keypoints(img, pose_result, skeleton, kpt_score_thr,
                         pose_kpt_color, pose_link_color, radius, thickness)

        if show:
            imshow(img, win_name, wait_time)

        if out_file is not None:
            imwrite(img, out_file)

        return img
