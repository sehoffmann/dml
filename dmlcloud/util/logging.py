import io
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist

import dmlcloud
from .. import slurm
from ..git import git_hash
from .thirdparty import is_imported, ML_MODULES, try_get_version


class IORedirector:
    """
    Context manager to redirect stdout and stderr to a file.
    Data is written to the file and the original streams.
    """

    # Caveats:
    #  * We always need to forward the current stdout/stderr. People can change them.
    #  * Even after uninstall, people can still hold reference to the redirected streams.
    #    Hence, we must be fault tolorant and not crash if the file is closed or the streams are changed.

    class Stdout:
        def __init__(self, parent):
            self.parent = parent

        def write(self, data):
            if self.parent.file is not None:
                self.parent.file.write(data)

            if sys.stdout is self:  # Avoid infinite recursion
                self.parent._org_stdout.write(data)
            else:
                sys.stdout.write(data)

        def flush(self):
            if self.parent.file is not None:
                self.parent.file.flush()

            if sys.stdout is self:  # Avoid infinite recursion
                self.parent._org_stdout.flush()
            else:
                sys.stdout.flush()

    class Stderr:
        def __init__(self, parent):
            self.parent = parent

        def write(self, data):
            if self.parent.file is not None:
                self.parent.file.write(data)

            if sys.stderr is self:  # Avoid infinite recursion
                self.parent._org_stderr.write(data)
            else:
                sys.stderr.write(data)

        def flush(self):
            if self.parent.file is not None:
                self.parent.file.flush()

            if sys.stderr is self:  # Avoid infinite recursion
                self.parent._org_stderr.flush()
            else:
                sys.stderr.flush()

    def __init__(self, log_file: Path):
        self.path = log_file
        self.file = None
        self._org_stdout = None
        self._org_stderr = None

    def install(self):
        if self.file is not None:
            return

        self.file = self.path.open('a', encoding='utf-8', errors='replace')
        self._org_stdout = sys.stdout
        self._org_stderr = sys.stderr
        self._org_stdout.flush()
        self._org_stderr.flush()

        sys.stdout = self.Stdout(self)
        sys.stderr = self.Stderr(self)

    def uninstall(self):
        if self.file is None:
            raise ValueError('IORedirector is not installed')

        sys.stdout = self._org_stdout
        sys.stderr = self._org_stderr

        self.file = None
        self._org_stdout = None
        self._org_stderr = None
        self.file.close()

    def __enter__(self):
        self.install()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.uninstall()


class DevNullIO(io.TextIOBase):
    """
    Dummy TextIOBase that will simply ignore anything written to it similar to /dev/null
    """

    def write(self, msg):
        pass


def experiment_header(
    name: str | None,
    checkpoint_dir: str | None,
    date: datetime,
) -> str:
    msg = f'...............  Experiment: {name if name else "N/A"}  ...............\n'
    msg += f'- Date: {date}\n'
    msg += f'- Checkpoint Dir: {checkpoint_dir if checkpoint_dir else "N/A"}\n'
    msg += f'- Training on {dist.get_world_size()} GPUs\n'
    return msg


def general_diagnostics() -> str:
    msg = '* GENERAL:\n'
    msg += f'    - argv: {sys.argv}\n'
    msg += f'    - cwd: {Path.cwd()}\n'

    msg += f'    - host (root): {os.environ.get("HOSTNAME")}\n'
    msg += f'    - user: {os.environ.get("USER")}\n'
    msg += f'    - git-hash: {git_hash()}\n'
    msg += f'    - conda-env: {os.environ.get("CONDA_DEFAULT_ENV", "N/A")}\n'
    msg += f'    - sys-prefix: {sys.prefix}\n'
    msg += f'    - backend: {dist.get_backend()}\n'
    msg += f'    - cuda: {torch.cuda.is_available()}\n'

    msg += '* VERSIONS:\n'
    msg += f'    - python: {sys.version}\n'
    msg += f'    - cuda (torch): {torch.version.cuda}\n'
    try:
        msg += '      - ' + Path('/proc/driver/nvidia/version').read_text().splitlines()[0] + '\n'
    except (FileNotFoundError, IndexError):
        pass

    msg += f'    - dmlcloud: {dmlcloud.__version__}\n'

    for module_name in ML_MODULES:
        if is_imported(module_name):
            msg += f'    - {module_name}: {try_get_version(module_name)}\n'

    if 'SLURM_JOB_ID' in os.environ:
        msg += '* SLURM:\n'
        msg += f'    - SLURM_JOB_ID = {slurm.slurm_job_id()}\n'
        msg += f'    - SLURM_STEP_ID = {slurm.slurm_step_id()}\n'
        msg += f'    - SLURM_STEP_NODELIST = {os.environ.get("SLURM_STEP_NODELIST")}\n'
        msg += f'    - SLURM_TASKS_PER_NODE = {os.environ.get("SLURM_TASKS_PER_NODE")}\n'
        msg += f'    - SLURM_STEP_GPUS = {os.environ.get("SLURM_STEP_GPUS")}\n'
        msg += f'    - SLURM_GPUS_ON_NODE = {os.environ.get("SLURM_GPUS_ON_NODE")}\n'
        msg += f'    - SLURM_CPUS_PER_TASK = {os.environ.get("SLURM_CPUS_PER_TASK")}'

    return msg
