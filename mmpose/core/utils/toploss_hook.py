from mmcv.runner import HOOKS, Hook

import numpy as np
from skimage import exposure
import cv2
import os
import torch

import mmcv
from mmpose.core.post_processing import oks_nms
from mmpose.core.visualization import imshow_keypoints

from torchvision import transforms
invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])

@HOOKS.register_module()
class TopLossHook(Hook):

    def __init__(self, top_k_top_losses):
        self.top_k_top_losses = top_k_top_losses
        self.top_k_max_train_loss = - float ("inf")
        self.top_k_max_val_loss = - float ("inf")
        self.top_k_max_train_loss_sample = []
        self.top_k_max_val_loss_sample = []

    def clear(self):
        self.top_k_max_train_loss = - float ("inf")
        self.top_k_max_val_loss = - float ("inf")
        self.top_k_max_train_loss_sample = []
        self.top_k_max_val_loss_sample = []

    def before_train_epoch(self, runner):
        self.clear()

    def after_train_iter(self, runner):
        # Update top loss sample
        self._get_top_loss_of_k_sample_in_batch(runner, subset='train')

    def after_train_epoch(self, runner):
        # Generate visualize image from top samples
        topk_results = self.top_k_max_train_loss_sample
        topk_joints = self._get_joints(runner.data_loader.dataset, topk_results)

        visualize_output = self._visualize_results(topk_results, topk_joints)
        runner.outputs['train_visualize_output'] = visualize_output

        topk_image = [os.path.basename(result['image_paths'][0]) for result in topk_results]
        epoch = runner.epoch+1
        log_str = f'Epoch(train) [{epoch}][{len(runner.data_loader)}]\t Top {self.top_k_top_losses} loss train images: ' + ', '.join(topk_image)
        runner.logger.info(log_str)
        self.clear()
        
    def before_val_epoch(self, runner):
        self.clear()

    def after_val_iter(self, runner):
        # Update top loss sample
        self._get_top_loss_of_k_sample_in_batch(runner, subset='val')

    def after_val_epoch(self, runner):
        # Generate visualize image from top samples
        topk_results = self.top_k_max_val_loss_sample
        topk_joints = self._get_joints(runner.data_loader.dataset, topk_results)

        visualize_output = self._visualize_results(topk_results, topk_joints)
        runner.outputs['val_visualize_output'] = visualize_output

        topk_image = [os.path.basename(result['image_paths'][0]) for result in topk_results]
        epoch = runner.epoch
        log_str = f'Epoch(val) [{epoch}][{len(runner.data_loader)}]\t Top {self.top_k_top_losses} loss val images: ' + ', '.join(topk_image)
        runner.logger.info(log_str)
        self.clear()
        
    def _get_top_loss_of_k_sample_in_batch(self, runner, subset='train'):
        # If the whole batch have loss smaller than self.top_k_max_loss
        if subset == 'train':
            top_k_max_loss = self.top_k_max_train_loss
            top_k_max_loss_sample = self.top_k_max_train_loss_sample
        elif subset == 'val':
            top_k_max_loss = self.top_k_max_val_loss
            top_k_max_loss_sample = self.top_k_max_val_loss_sample
        else:
            raise ValueError(f"Subset {subset} are not defined")

        # Only care the sample that have loss bigger than top_k_max_loss
        kept_sample = []
        results = runner.outputs["results"]
        for i in range(len(results)):
            loss = results[i]['loss']

            if loss > top_k_max_loss:
                kept_sample.append(results[i])

        # Sort them by loss and get top k value
        kept_sample = sorted(kept_sample, key=lambda x: x['loss'], reverse=True)[:self.top_k_top_losses]
        if len(kept_sample) > 0:
            # Sort the current top sample, and kept top k only
            top_k_max_loss_sample.extend(kept_sample)
            top_k_max_loss_sample.sort(key=lambda x: x['loss'], reverse=True)
            top_k_max_loss_sample = top_k_max_loss_sample[:self.top_k_top_losses]

            # Get the min loss from the top k sample to update top_k_max_loss
            min_loss = top_k_max_loss_sample[-1]['loss']
        
            if subset == 'train':
                self.top_k_max_train_loss = min_loss
                self.top_k_max_train_loss_sample = top_k_max_loss_sample
            elif subset == 'val':
                self.top_k_max_val_loss = min_loss
                self.top_k_max_val_loss_sample = top_k_max_loss_sample
    
    def _get_joints(self, dataset, top_k_loss_samples):
        """Get target joints"""
        joints = []
        coco = dataset.coco
        for sample in top_k_loss_samples:
            w_scale, h_scale = sample['rescale']
            image_name = os.path.basename(sample['image_paths'][0])
            img_id = dataset.name2id[image_name]
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anno = coco.loadAnns(ann_ids)
            joint = dataset._get_joints(anno)
            # Rescale joints to input image size
            joint[:,:,0] = joint[:,:,0] / w_scale
            joint[:,:,1] = joint[:,:,1] / h_scale
            joints.append(joint)

        return joints

    def _draw_target(self, img, joints):
        """Draw target joints"""
        img_copy = img.copy()
        colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [128, 128, 128]]
        for i in range(joints.shape[0]): # Num card in image
            for j in range(joints.shape[1]): # Num keypoints
                x = int(joints[i, j, 0])
                y = int(joints[i, j, 1])
                img_copy = cv2.circle(img_copy, (x, y),
                                        radius=4, color=colors[j], thickness=5)
        return img_copy

    def _draw_text(self, img, text,
                    font=cv2.FONT_HERSHEY_PLAIN,
                    font_scale=1,
                    font_thickness=1,
                    text_color=(0, 0, 0),
                    text_color_bg=(255, 255, 255)):
        img_copy = img.copy()
        h, w, _ = img.shape
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_w, text_h = text_size
        x = w - 20 - text_w
        y = 10
        img_copy = cv2.rectangle(img_copy, (x,y), (x + text_w + 10, y + text_h + 10), text_color_bg, -1)
        img_copy = cv2.putText(img_copy, text, (x+5, y+5 + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)
        return img_copy

    def _add_figure_title(self, visualize_output):
        num_top_k, num_img, h, w, _ = visualize_output.shape
        ret_output = np.zeros((num_top_k, h+50, w*num_img, 3), dtype=np.uint8)
        font=cv2.FONT_HERSHEY_PLAIN
        font_scale=2
        font_thickness=1
        text_color = (0, 0, 0)
        texts = ['keypoint target', 'keypoint prediction', 
                    'top_left heatmap', 'bottom_left heatmap', 'bottom_right heatmap', 'top_right heatmap']
        for i in range(num_top_k):
            for j in range(num_img):
                text_size, _ = cv2.getTextSize(texts[j], font, font_scale, font_thickness)
                text_w, text_h = text_size
                new_img = np.zeros((h+50, w, 3), dtype=np.uint8)
                new_img[:50, :] = 255
                new_img[50:, :] = visualize_output[i, j]
                new_img = cv2.putText(new_img, texts[j], (int((w-text_w)/2), 40-text_h+font_scale-1), 
                                        font, font_scale, text_color, font_thickness)
                ret_output[i, :, j*w:(j+1)*w, :] = new_img
        return ret_output

    def _visualize_heatmap(self, img, heatmap):
        """Draw heatmap images, one heatmap per keypoint"""
        img = mmcv.imread(img)
        img = img.copy()
        heatmap_embedded_img = []
        for slice in heatmap:
            img_copy = img.copy()
            map_img = exposure.rescale_intensity(slice, out_range=(0, 255))
            map_img = np.uint8(map_img) 
            heatmap_img = cv2.applyColorMap(map_img, cv2.COLORMAP_JET)
            heatmap_embedded_img.append(cv2.addWeighted(heatmap_img, 0.7, img_copy, 0.3, 0))

        heatmap_embedded_img = np.array(heatmap_embedded_img)
        return heatmap_embedded_img

    def _visualize_keypoint(self, pose_results, img):
        img_copy = img.copy()
        img_copy = self._show_result(
            img_copy,
            pose_results,
        )

        return img_copy

    def _visualize_results(self, results, joints):
        img_keypoint = []
        img_heatmap = []
        target_img = []
        for i, result in enumerate(results):
            pose_results = []
            img = result['image']
            img = invTrans(img).detach().cpu().numpy()
            img = np.transpose(img, (1,2,0))
            img = (img*255).astype(np.uint8)
            
            img_heatmap.append(self._visualize_heatmap(img, result['output_heatmap'][0]))
            target_img.append(self._draw_target(img, joints[i]))
            
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
                keep = oks_nms(pose_results, thr=0.9, sigmas=[0.025, 0.025, 0.025, 0.025])
                pose_results = [pose_results[_keep] for _keep in keep]
                img = self._visualize_keypoint(pose_results, img)
            img = self._draw_text(img, f'loss: {result["loss"]:.4f}')
            img_keypoint.append(img)
        img_keypoint = np.array(img_keypoint) \
            .reshape(len(results), 1, img_keypoint[0].shape[0], img_keypoint[0].shape[1], img_keypoint[0].shape[2])
        img_heatmap = np.array(img_heatmap)
        target_img = np.array(target_img) \
            .reshape(len(results), 1, target_img[0].shape[0], target_img[0].shape[1], target_img[0].shape[2])
        visualize_output = np.concatenate((target_img, img_keypoint, img_heatmap), axis=1)
        visualize_output = self._add_figure_title(visualize_output)

        return visualize_output

    def _show_result(self,
                    img,
                    result,
                    kpt_score_thr=0.3,
                    radius=4,
                    thickness=1):
        skeleton = [[0, 1], [1, 2], [2, 3], [3, 0]]
        pose_link_color = [[128, 255, 0], [0, 255, 128], [255, 128, 0], [255, 128, 128]]
        pose_kpt_color = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [128, 128, 128]]
        pose_result = []
        for res in result:
            pose_result.append(res['keypoints'])

        imshow_keypoints(img, pose_result, skeleton, kpt_score_thr,
                         pose_kpt_color, pose_link_color, radius, thickness)

        return img