import sys

import dmlcloud as dml
import pytest


@dml.root_only
def return_root_rank():
    """TEST_DOC_STRING"""
    return dml.rank()


@dml.root_only
class RootOnlyStage(dml.Stage):
    """TEST_DOC_STRING"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cb_executed = {
            'pre_stage': False,
            'post_stage': False,
            'pre_epoch': False,
            'post_epoch': False,
            'run_epoch': False,
        }

    def pre_stage(self):
        """TEST_DOC_STRING"""
        self.cb_executed['pre_stage'] = True

    def post_stage(self):
        """TEST_DOC_STRING"""
        self.cb_executed['post_stage'] = True

    def pre_epoch(self):
        """TEST_DOC_STRING"""
        self.cb_executed['pre_epoch'] = True

    def post_epoch(self):
        """TEST_DOC_STRING"""
        self.cb_executed['post_epoch'] = True

    def run_epoch(self):
        """TEST_DOC_STRING"""
        self.cb_executed['run_epoch'] = True


class PartialRootOnlyStage(dml.Stage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cb_executed = {
            'pre_stage': False,
            'post_stage': False,
            'pre_epoch': False,
            'post_epoch': False,
            'run_epoch': False,
        }

    def pre_stage(self):
        self.cb_executed['pre_stage'] = True

    @dml.root_only
    def post_stage(self):
        """TEST_DOC_STRING"""
        self.cb_executed['post_stage'] = True

    @dml.root_only
    def pre_epoch(self):
        """TEST_DOC_STRING"""
        self.cb_executed['pre_epoch'] = True

    def post_epoch(self):
        self.cb_executed['post_epoch'] = True

    @dml.root_only
    def run_epoch(self):
        self.cb_executed['run_epoch'] = True


@dml.root_only
class RootOnlyPipeline(dml.Pipeline):
    """TEST_DOC_STRING"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cb_executed = {
            'pre_run': False,
            'post_run': False,
        }

    def pre_run(self):
        """TEST_DOC_STRING"""
        self.cb_executed['pre_run'] = True

    def post_run(self):
        """TEST_DOC_STRING"""
        self.cb_executed['post_run'] = True


class PartialRootOnlyPipeline(dml.Pipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cb_executed = {
            'pre_run': False,
            'post_run': False,
        }

    @dml.root_only
    def pre_run(self):
        """TEST_DOC_STRING"""
        self.cb_executed['pre_run'] = True

    def post_run(self):
        self.cb_executed['post_run'] = True


class TestRootOnly:
    def test_function(self, distributed_environment):
        ranks = distributed_environment(4).start(return_root_rank)
        assert ranks == [0, None, None, None]
        assert return_root_rank.__name__ == 'return_root_rank'
        assert return_root_rank.__doc__ == 'TEST_DOC_STRING'

    @staticmethod
    def _test_stage_run():
        stage = RootOnlyStage(epochs=1)
        pipe = dml.Pipeline()
        pipe.append(stage)
        pipe.run()
        return stage.cb_executed

    def test_stage(self, distributed_environment):
        results = distributed_environment(3).start(TestRootOnly._test_stage_run)

        assert [r['pre_stage'] for r in results] == [True, False, False]
        assert [r['post_stage'] for r in results] == [True, False, False]
        assert [r['pre_epoch'] for r in results] == [True, False, False]
        assert [r['post_epoch'] for r in results] == [True, False, False]
        assert [r['run_epoch'] for r in results] == [True, False, False]

        assert RootOnlyStage.__name__ == 'RootOnlyStage'
        assert RootOnlyStage.__doc__ == 'TEST_DOC_STRING'

        assert RootOnlyStage.pre_stage.__name__ == 'pre_stage'
        assert RootOnlyStage.pre_stage.__doc__ == 'TEST_DOC_STRING'

        assert RootOnlyStage.post_stage.__name__ == 'post_stage'
        assert RootOnlyStage.post_stage.__doc__ == 'TEST_DOC_STRING'

        assert RootOnlyStage.pre_epoch.__name__ == 'pre_epoch'
        assert RootOnlyStage.pre_epoch.__doc__ == 'TEST_DOC_STRING'

        assert RootOnlyStage.post_epoch.__name__ == 'post_epoch'
        assert RootOnlyStage.post_epoch.__doc__ == 'TEST_DOC_STRING'

        assert RootOnlyStage.run_epoch.__name__ == 'run_epoch'
        assert RootOnlyStage.run_epoch.__doc__ == 'TEST_DOC_STRING'

    @staticmethod
    def _test_partial_stage_run():
        stage = PartialRootOnlyStage(epochs=1)
        pipe = dml.Pipeline()
        pipe.append(stage)
        pipe.run()
        return stage.cb_executed

    def test_partial_stage(self, distributed_environment):
        results = distributed_environment(3).start(TestRootOnly._test_partial_stage_run)

        assert [r['pre_stage'] for r in results] == [True, True, True]
        assert [r['post_stage'] for r in results] == [True, False, False]
        assert [r['pre_epoch'] for r in results] == [True, False, False]
        assert [r['post_epoch'] for r in results] == [True, True, True]
        assert [r['run_epoch'] for r in results] == [True, False, False]

        assert PartialRootOnlyStage.post_stage.__name__ == 'post_stage'
        assert PartialRootOnlyStage.post_stage.__doc__ == 'TEST_DOC_STRING'

        assert PartialRootOnlyStage.pre_epoch.__name__ == 'pre_epoch'
        assert PartialRootOnlyStage.pre_epoch.__doc__ == 'TEST_DOC_STRING'

    @staticmethod
    def _test_pipeline_run():
        pipe = RootOnlyPipeline()
        pipe.append(RootOnlyStage(epochs=1))
        pipe.run()
        return pipe.cb_executed

    def test_pipeline(self, distributed_environment):
        results = distributed_environment(3).start(TestRootOnly._test_pipeline_run)

        assert [r['pre_run'] for r in results] == [True, False, False]
        assert [r['post_run'] for r in results] == [True, False, False]

        assert RootOnlyPipeline.__name__ == 'RootOnlyPipeline'
        assert RootOnlyPipeline.__doc__ == 'TEST_DOC_STRING'

        assert RootOnlyPipeline.pre_run.__name__ == 'pre_run'
        assert RootOnlyPipeline.pre_run.__doc__ == 'TEST_DOC_STRING'

        assert RootOnlyPipeline.post_run.__name__ == 'post_run'
        assert RootOnlyPipeline.post_run.__doc__ == 'TEST_DOC_STRING'

    @staticmethod
    def _test_partial_pipeline_run():
        pipe = PartialRootOnlyPipeline()
        pipe.append(RootOnlyStage(epochs=1))
        pipe.run()
        return pipe.cb_executed

    def test_partial_pipeline(self, distributed_environment):
        results = distributed_environment(3).start(TestRootOnly._test_partial_pipeline_run)

        assert [r['pre_run'] for r in results] == [True, False, False]
        assert [r['post_run'] for r in results] == [True, True, True]

        assert PartialRootOnlyPipeline.pre_run.__name__ == 'pre_run'
        assert PartialRootOnlyPipeline.pre_run.__doc__ == 'TEST_DOC_STRING'


if __name__ == '__main__':
    sys.exit(pytest.main([__file__]))
