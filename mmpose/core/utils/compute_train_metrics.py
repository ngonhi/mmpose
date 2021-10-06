# Copyright (c) OpenMMLab. All rights reserved.
import tempfile
import warnings

from mmcv.runner import HOOKS, Hook
from mmpose import apis
# from mmpose.apis import single_gpu_test

@HOOKS.register_module()
class ComputeTrainMetricsHook(Hook):
    def __init__(self,
                 dataloader,
                 **eval_kwargs):
        self.dataloader = dataloader
        self.eval_kwargs = eval_kwargs

    def after_train_epoch(self, runner):
        """Called after every training epoch to evaluate the results."""
        results = apis.single_gpu_test(runner.model, self.dataloader)
        runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
        self.evaluate(runner, results)

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

        for name, val in eval_res.items():
            runner.log_buffer.output[name+'_train'] = val
        runner.log_buffer.ready = True

        return None
