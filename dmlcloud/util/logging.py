import io
import os
import subprocess
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

    class Stdout:
        def __init__(self, parent):
            self.parent = parent

        def write(self, data):
            self.parent.file.write(data)
            self.parent.stdout.write(data)

        def flush(self):
            self.parent.file.flush()
            self.parent.stdout.flush()

    class Stderr:
        def __init__(self, parent):
            self.parent = parent

        def write(self, data):
            self.parent.file.write(data)
            self.parent.stderr.write(data)

        def flush(self):
            self.parent.file.flush()
            self.parent.stderr.flush()

    def __init__(self, log_file: Path):
        self.path = log_file
        self.file = None
        self.stdout = None
        self.stderr = None

    def install(self):
        if self.file is not None:
            return

        self.file = self.path.open('a')
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        self.stdout.flush()
        self.stderr.flush()

        sys.stdout = self.Stdout(self)
        sys.stderr = self.Stderr(self)

    def uninstall(self):
        self.stdout.flush()
        self.stderr.flush()

        sys.stdout = self.stdout
        sys.stderr = self.stderr

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

    if torch.cuda.is_available():
        msg += '* GPUs (root):\n'
        nvsmi = subprocess.run(['nvidia-smi', '-L'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.decode()
        for line in nvsmi.splitlines():
            msg += f'    - {line}\n'

    msg += '* VERSIONS:\n'
    msg += f'    - python: {sys.version}\n'
    msg += f'    - dmlcloud: {dmlcloud.__version__}\n'
    msg += f'    - cuda: {torch.version.cuda}\n'
    try:
        msg += '      - ' + Path('/proc/driver/nvidia/version').read_text().splitlines()[0] + '\n'
    except (FileNotFoundError, IndexError):
        pass

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
