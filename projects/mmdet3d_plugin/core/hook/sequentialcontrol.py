# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner.hooks import HOOKS, Hook
from .utils import is_parallel

__all__ = ['SequentialControlHook']


@HOOKS.register_module()
class SequentialControlHook(Hook):
    """ """

    def __init__(self, temporal_start_iter=-1):
        super().__init__()
        self.temporal_start_iter = temporal_start_iter
        self.temporal = False

    def set_temporal_flag(self, runner, flag):
        if is_parallel(runner.model.module):
            runner.model.module.module.do_history = flag
            if flag:
                runner.model.module.module.pts_bbox_head.code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            else:
                runner.model.module.module.pts_bbox_head.code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]
        else:
            runner.model.module.do_history = flag
            if flag:
                runner.model.module.pts_bbox_head.code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            else:
                runner.model.module.pts_bbox_head.code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]

    def before_run(self, runner):
        self.set_temporal_flag(runner, False)

    def after_train_iter(self, runner):
        if runner.iter >= self.temporal_start_iter and not self.temporal:
            self.set_temporal_flag(runner, True)
            self.temporal = True