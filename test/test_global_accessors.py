import dmlcloud as dml


class DummyStage(dml.Stage):
    def run_epoch(self):
        pass


class ProbingCallback(dml.Callback):

    def __init__(self, pipe=None, stage=None):
        self.pipe = pipe
        self.stage = stage
        self.pipe_test = False
        self.stage_test = False


    def pre_run(self, pipe):
        self.pipe_test = dml.current_pipe() is self.pipe

    def pre_stage(self, stage):
        self.stage_test = dml.current_stage() is self.stage


class LogCallback(dml.Callback):

    def __init__(self):
        self. i = 0

    def pre_epoch(self, stage):
        dml.log_metric('test', self.i)
        self.i += 1


class TestGlobalAccessors:
    def test_accessors(self, torch_distributed):
        pipe = dml.Pipeline()
        stage1 = DummyStage()
        stage2 = DummyStage()
        pipe.append(stage1)
        pipe.append(stage2)


        cb1 = ProbingCallback(pipe)
        cb2 = ProbingCallback(stage = stage1)
        cb3 = ProbingCallback(stage = stage2)

        pipe.add_callback(cb1)
        stage1.add_callback(cb2)
        stage2.add_callback(cb3)


        assert dml.current_pipe() is None
        assert dml.current_stage() is None
        
        pipe.run()
        assert cb1.pipe_test
        assert cb2.stage_test
        assert cb3.stage_test

        assert dml.current_pipe() is None
        assert dml.current_stage() is None

    
    def test_logging(self, torch_distributed):
        pipe = dml.Pipeline()
        stage1 = DummyStage(epochs=3)
        stage2 = DummyStage(epochs=1)
        pipe.append(stage1)
        pipe.append(stage2)

        pipe.add_callback(LogCallback())

        pipe.run()

        assert 'test' in stage1.history
        assert list(stage1.history['test']) == [0,1,2]
        
        assert 'test' in stage2.history
        assert list(stage2.history['test']) == [3]




if __name__ == '__main__':
    sys.exit(pytest.main([__file__]))
