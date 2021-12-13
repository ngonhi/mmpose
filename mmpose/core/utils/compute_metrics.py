# Copyright (c) OpenMMLab. All rights reserved.
import tempfile
import warnings

from mmcv.runner import HOOKS, Hook

@HOOKS.register_module()
class ComputeMetricsHook(Hook):
    def __init__(self,
                 dataloaders,
                 eval_train_kwargs,
                 eval_val_kwargs):
        self.train_dataloader = dataloaders[0]
        self.val_dataloader = dataloaders[1]
        self.train_eval_kwargs = eval_train_kwargs
        self.val_eval_kwargs = eval_val_kwargs
        self.val_results = []
        self.sum = {}
        self.count = 0
        self.avg = {}

    def reset(self):
        """Reset the internal evaluation results."""
        self.sum = {}
        self.count = 0
        self.avg = {}
        self.val_results = []

    def before_train_epoch(self, runner):
        self.reset()

    def after_train_iter(self, runner):
        """Called after every training iteration to aggregate the results."""
        results = runner.outputs['results']
        temp = [None]*len(results)
        for n, result in enumerate(results):
            temp[n] = result.copy()
            temp[n]['preds'] = result['preds'].copy()
            for j, item in enumerate(result['preds']):
                w_scale, h_scale = result['rescale']
                temp[n]['preds'][j] = item.copy()
                temp[n]['preds'][j][:, 0] = item[:, 0] * w_scale
                temp[n]['preds'][j][:, 1] = item[:, 1] * h_scale
        eval_res = self.evaluate(runner, temp, subset='train')
        n = len(results)
        self.update(eval_res, n)

    def after_train_epoch(self, runner):
        """Called after every training epoch to evaluate the results."""
        runner.log_buffer.output['eval_iter_num'] = len(self.train_dataloader)
        for name, val in self.avg.items():
            runner.log_buffer.output[name+'_train'] = val
        runner.log_buffer.ready = True
        self.reset()

    def before_val_epoch(self, runner):
        self.reset()

    def after_val_iter(self, runner):
        """Called after every validation iteration to aggregate the results."""
        results = runner.outputs['results']
        temp = [None]*len(results)
        for n, result in enumerate(results):
            temp[n] = result.copy()
            temp[n]['preds'] = result['preds'].copy()
            for j, item in enumerate(result['preds']):
                w_scale, h_scale = result['rescale']
                temp[n]['preds'][j] = item.copy()
                temp[n]['preds'][j][:, 0] = item[:, 0] * w_scale
                temp[n]['preds'][j][:, 1] = item[:, 1] * h_scale
        self.val_results += [{
            'preds': t['preds'],
            'scores': t['scores'],
            'image_paths': t['image_paths'],
            'output_heatmap': None
        } for t in temp]

    def after_val_epoch(self, runner):
        """Called after every validation epoch to evaluate the results."""
        runner.log_buffer.output['eval_iter_num'] = len(self.val_dataloader)
        eval_res = self.evaluate(runner, self.val_results, subset='val')
        for name, val in eval_res.items():
            runner.log_buffer.output[name+'_val'] = val
        runner.log_buffer.ready = True
        self.reset()

    def evaluate(self, runner, results, subset='train'):
        """Evaluate the results.

        Args:
            runner (:obj:`mmcv.Runner`): The underlined training runner.
            results (list): Output results.
        """
        if subset == 'train':
            dataloader = self.train_dataloader
            eval_kwargs = self.train_eval_kwargs
        elif subset == 'val':
            dataloader = self.val_dataloader
            eval_kwargs = self.val_eval_kwargs
        with tempfile.TemporaryDirectory() as tmp_dir:
            eval_res = dataloader.dataset.evaluate(
                results,
                res_folder=tmp_dir,
                logger=runner.logger,
                **eval_kwargs)
        return eval_res

    def update(self, eval_res, n):
        self.count += n
        for name, val in eval_res.items():
            if name not in self.sum:       
                self.sum[name] = val*n
            else:
                self.sum[name] += val*n
        
            self.avg[name] = self.sum[name]/self.count
            
