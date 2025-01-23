import sys

import pytest

from dmlcloud.util.logging import IORedirector


class DummyStream:

    def __init__(self, stdout=True):
        self.data = ''
        self.is_stdout = stdout

    def write(self, data):
        self.data += data

    def flush(self):
        pass

    def __enter__(self):
        self._org_stream = getattr(sys, 'stdout' if self.is_stdout else 'stderr')
        setattr(sys, 'stdout' if self.is_stdout else 'stderr', self)
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        setattr(sys, 'stdout' if self.is_stdout else 'stderr', self._org_stream)


class TestIORedirector:
    def test_context_manager(self, tmp_path):
        org_stdout = sys.stdout
        org_stderr = sys.stderr

        with IORedirector(tmp_path / 'log.txt'):
            assert sys.stdout is not org_stdout
            assert sys.stderr is not org_stderr

        assert sys.stdout is org_stdout
        assert sys.stderr is org_stderr


    def test_file_creation(self, tmp_path):
        with IORedirector(tmp_path / 'log.txt'):
            pass
        assert (tmp_path / 'log.txt').exists()
        assert (tmp_path / 'log.txt').read_text() == ''


    def test_basic_write(self, tmp_path):
        with DummyStream() as out, DummyStream(stdout=False) as err:
            with IORedirector(tmp_path / 'log.txt'):
                print('Hello, world!')
                print('Error message', file=sys.stderr)

        file_content = (tmp_path / 'log.txt').read_text()
        assert file_content == 'Hello, world!\nError message\n'

        assert out.data == 'Hello, world!\n'
        assert err.data == 'Error message\n'

        assert sys.stdout is out._org_stream
        assert sys.stderr is err._org_stream

    
    def test_writes_after_exit(self, tmp_path):
        with DummyStream() as out, DummyStream(stdout=False) as err:
            with IORedirector(tmp_path / 'log.txt'):
                saved_out = sys.stdout
                saved_err = sys.stderr

            print('Test', file=saved_out)
            print('Error', file=saved_err)
            assert out.data == 'Test\n'
            assert err.data == 'Error\n'

            file_content = (tmp_path / 'log.txt').read_text()
            assert file_content == ''

        # Now we reset and replace sys.stdout and sys.stderr again, writes should go to the new streams
        with DummyStream() as out2, DummyStream(stdout=False) as err2:
            print('Test', file=saved_out)
            print('Error', file=saved_err)
            
            assert out.data == 'Test\n'
            assert err.data == 'Error\n'

            assert out2.data == 'Test\n'
            assert err2.data == 'Error\n'


if __name__ == '__main__':
    sys.exit(pytest.main([__file__]))
