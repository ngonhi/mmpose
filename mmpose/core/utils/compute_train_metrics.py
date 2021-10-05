# Copyright (c) OpenMMLab. All rights reserved.
import tempfile
import warnings

from mmcv.runner import EvalHook as _EvalHook

MMPOSE_GREATER_KEYS = [
    'acc', 'ap', 'ar', 'pck', 'auc', '3dpck', 'p-3dpck', '3dauc', 'p-3dauc'
]
MMPOSE_LESS_KEYS = ['loss', 'epe', 'nme', 'mpjpe', 'p-mpjpe', 'n-mpjpe']


class ComputeTrainMetricsHook(_EvalHook):
    def __init__(self,
                 dataloader,
                 start=None,
                 interval=1,
                 by_epoch=True,
                 save_best=None,
                 rule=None,
                 test_fn=None,
                 greater_keys=MMPOSE_GREATER_KEYS,
                 less_keys=MMPOSE_LESS_KEYS,
                 **eval_kwargs):

        if test_fn is None:
            from mmpose.apis import single_gpu_test
            test_fn = single_gpu_test

        super().__init__(dataloader, start, interval, by_epoch, save_best,
                         rule, test_fn, greater_keys, less_keys, **eval_kwargs)

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
