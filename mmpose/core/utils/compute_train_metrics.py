import tempfile

from mmcv.runner import HOOKS, Hook

@HOOKS.register_module()
class ComputeTrainMetricsHook(Hook):
    def __init__(self, dataloader):
        self.results = []
        self.dataloader = dataloader

    def clear(self):
        self.results = []

    def before_train_epoch(self, runner):
        self.clear()

    # Compute metrics after epoch
    def after_train_iter(self, runner):
        result = runner.outputs['results']
        saved_result = []
        # Rescale to original dataset scale
        for idx, sample in enumerate(result):
            saved_sample = {}
            saved_sample['scores'] = sample['scores']
            saved_sample['image_paths'] = sample['image_paths']
            saved_sample['preds'] = sample['preds']
            rescale = sample['rescale']
            if len(sample['preds']) > 0 and rescale:
                for i, item in enumerate(saved_sample['preds']):
                    saved_sample['preds'][i][:, 0] = saved_sample['preds'][i][:, 0] * rescale[0]
                    saved_sample['preds'][i][:, 1] = saved_sample['preds'][i][:, 1] * rescale[1]
            saved_result.append(saved_sample)
        self.results += saved_result

    def after_train_epoch(self, runner):
        # Compute metrics
        with tempfile.TemporaryDirectory() as tmp_dir:
            eval_res = self.dataloader.dataset.evaluate(
                self.results,
                res_folder=tmp_dir,
                logger=runner.logger)
        
        for name, val in eval_res.items():
            runner.log_buffer.output[name+'_train'] = val
        self.clear()
