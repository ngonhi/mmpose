# Copyright (c) OpenMMLab. All rights reserved.
import tempfile
import warnings

from mmcv.runner import HOOKS, Hook

@HOOKS.register_module()
class ComputeTrainMetricsHook(Hook):
    def __init__(self,
                 dataloader,
                 **eval_kwargs):
        self.dataloader = dataloader
        self.eval_kwargs = eval_kwargs
        self.sum = {}
        self.count = 0
        self.avg = {}

    def reset(self):
        """Reset the internal evaluation results."""
        self.sum = {}
        self.count = 0
        self.avg = {}

    def before_train_epoch(self, runner):
        self.reset()

    def after_train_iter(self, runner):
        """Called after every training iteration to aggregate the results."""
        results = runner.outputs['results']
        for n, result in enumerate(results):
            for j, item in enumerate(result['preds']):
                w_scale, h_scale = result['rescale']
                item[:, 0] = item[:, 0] * w_scale
                item[:, 1] = item[:, 1] * h_scale
                result['preds'][j] = item
            results[n] = result
        eval_res = self.evaluate(runner, results)
        n = len(results)
        self.update(eval_res, n)

    def after_train_epoch(self, runner):
        """Called after every training epoch to evaluate the results."""
        runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
        for name, val in self.avg.items():
            runner.log_buffer.output[name+'_train'] = val
        runner.log_buffer.ready = True
        self.reset()

    def evaluate(self, runner, results):
        """Evaluate the results.

        Args:
            runner (:obj:`mmcv.Runner`): The underlined training runner.
            results (list): Output results.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            eval_res = self.dataloader.dataset.evaluate(
                results,
                res_folder=tmp_dir,
                logger=runner.logger,
                **self.eval_kwargs)
        return eval_res

    def update(self, eval_res, n):
        self.count += n
        for name, val in eval_res.items():
            if name not in self.sum:       
                self.sum[name] = val*n
            else:
                self.sum[name] += val*n
        
            self.avg[name] = self.sum[name]/self.count
            
