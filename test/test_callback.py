import sys
import time

import dmlcloud as dml
from dmlcloud.core.callbacks import CallbackList
import pytest


class DummyCallback(dml.Callback):
    
    def __init__(self, idx):
        super().__init__()
        self.idx = idx

        self.t_pre_run = []
        self.t_post_run = []
        self.t_pre_stage = []
        self.t_post_stage = []
        self.t_cleanup = []
        self.t_pre_epoch = []
        self.t_post_epoch = []

    def pre_run(self, pipe):
        self.t_pre_run.append(time.time())
    
    def post_run(self, pipe):
        self.t_post_run.append(time.time())

    def pre_stage(self, stage):
        self.t_pre_stage.append(time.time())

    def post_stage(self, stage):
        self.t_post_stage.append(time.time())

    def cleanup(self, pipe, exc_type, exc_value, traceback):
        self.t_cleanup.append(time.time())

    def pre_epoch(self, stage):
        self.t_pre_epoch.append(time.time())
    
    def post_epoch(self, stage):
        self.t_post_epoch.append(time.time())


class DummyStage(dml.Stage):

    def __init__(self, name, epochs):
        super().__init__(name, epochs)
        self.t_pre_stage = []
        self.t_post_stage = []
        self.t_pre_epoch = []
        self.t_post_epoch = []
    
    def pre_stage(self):
        self.t_pre_stage.append(time.time())
    
    def post_stage(self):
        self.t_post_stage.append(time.time())

    def pre_epoch(self):
        self.t_pre_epoch.append(time.time())
    
    def post_epoch(self):
        self.t_post_epoch.append(time.time())

    def run_epoch(self):
        pass



class TestCallbackList:
    def test_priorities(self):
        cb_list = CallbackList()
        cb_list.append(DummyCallback(0), 100)
        cb_list.append(DummyCallback(1), 50)
        cb_list.append(DummyCallback(2), 200)
        cb_list.append(DummyCallback(3), -100)
        cb_list.append(DummyCallback(4), 100)

        indices = [cb.idx for cb in cb_list]
        assert indices == [3, 1, 0, 4, 2]

    def test_combining(self):
        cb_list1 = CallbackList()
        cb_list1.append(DummyCallback(0), 100)
        cb_list1.append(DummyCallback(1), 50)

        cb_list2 = CallbackList()
        cb_list2.append(DummyCallback(2), 200)
        cb_list2.append(DummyCallback(3), -100)
        cb_list2.append(DummyCallback(4), 50)

        combined1 = cb_list1 + cb_list2
        indices = [cb.idx for cb in combined1]
        assert indices == [3, 1, 4, 0, 2]

        # Order for same-priority depends on the order of the operands
        combined2 = cb_list2 + cb_list1
        indices = [cb.idx for cb in combined2]
        assert indices == [3, 4, 1, 0, 2]

    
    def test_len(self):
        cb_list = CallbackList()
        assert len(cb_list) == 0

        cb_list.append(DummyCallback(0), 100)
        assert len(cb_list) == 1

        cb_list.append(DummyCallback(1), 50)
        assert len(cb_list) == 2

        cb_list.append(DummyCallback(2), 200)
        assert len(cb_list) == 3


class TestCallback:

    def test_stage_methods(self, torch_distributed):
        pipe = dml.Pipeline()
        stage1 = DummyStage('stage1', 2)
        pipe.append(stage1)
        pipe.run()

        assert len(stage1.t_pre_stage) == 1
        assert len(stage1.t_post_stage) == 1
        assert len(stage1.t_pre_epoch) == 2
        assert len(stage1.t_post_epoch) == 2

        assert stage1.t_pre_stage[0] < stage1.t_pre_epoch[0]
        assert stage1.t_pre_epoch[0] < stage1.t_post_epoch[0]
        assert stage1.t_post_epoch[0] < stage1.t_pre_epoch[1]
        assert stage1.t_pre_epoch[1] < stage1.t_post_epoch[1]
        assert stage1.t_post_epoch[1] < stage1.t_post_stage[0]
    
    def test_stage_callback(self, torch_distributed):
        pipe = dml.Pipeline()
        stage1 = DummyStage('stage1', 1)
        stage2 = DummyStage('stage2', 1)
        cb = DummyCallback(0)
        
        pipe.append(stage1)
        pipe.append(stage2)
        
        stage1.add_callback(cb)

        pipe.run()

        assert len(cb.t_pre_stage) == 1
        assert len(cb.t_post_stage) == 1
        assert len(cb.t_pre_epoch) == 1
        assert len(cb.t_post_epoch) == 1
        assert len(cb.t_pre_run) == 0
        assert len(cb.t_post_run) == 0

        assert stage1.t_pre_stage[0] < cb.t_pre_stage[0]
        assert stage1.t_post_stage[0] < cb.t_post_stage[0]

    def test_stage_callback_priority(self, torch_distributed):
        pipe = dml.Pipeline()
        stage1 = DummyStage('stage1', 1)
        stage2 = DummyStage('stage2', 1)
        cb = DummyCallback(0)
        
        pipe.append(stage1)
        pipe.append(stage2)
        
        stage1.add_callback(cb, priority=-1)

        pipe.run()

        assert len(cb.t_pre_stage) == 1
        assert len(cb.t_post_stage) == 1
        assert len(cb.t_pre_epoch) == 1
        assert len(cb.t_post_epoch) == 1
        assert len(cb.t_pre_run) == 0
        assert len(cb.t_post_run) == 0

        assert cb.t_pre_stage[0] < stage1.t_pre_stage[0]
        assert cb.t_post_stage[0] < stage1.t_post_stage[0]


    def test_pipeline_callback(self, torch_distributed):
        pipe = dml.Pipeline()
        stage1 = DummyStage('stage1', 1)
        stage2 = DummyStage('stage2', 1)
        cb = DummyCallback(0)
        
        pipe.append(stage1)
        pipe.append(stage2)
        pipe.add_callback(cb)

        pipe.run()

        assert len(cb.t_pre_run) == 1
        assert len(cb.t_post_run) == 1
        assert len(cb.t_cleanup) == 1
        assert len(cb.t_pre_stage) == 2
        assert len(cb.t_post_stage) == 2
        assert len(cb.t_pre_epoch) == 2
        assert len(cb.t_post_epoch) == 2

        assert cb.t_pre_run[0] < cb.t_pre_stage[0]
        assert cb.t_post_stage[0] < cb.t_post_run[0]
        assert cb.t_post_run[0] < cb.t_cleanup[0]


if __name__ == '__main__':
    sys.exit(pytest.main([__file__]))
