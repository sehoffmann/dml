import datetime
import logging
import secrets
from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf

from dmlcloud.slurm import slurm_job_id


def sanitize_filename(filename: str) -> str:
    return filename.replace('/', '_')


def generate_id() -> str:
    s = secrets.token_urlsafe(5)
    return s.replace('-', 'a').replace('_', 'b')


def generate_checkpoint_path(
    root: Path | str, name: Optional[str] = None, creation_time: Optional[datetime.datetime] = None
) -> Path:
    root = Path(root)

    if name is None:
        name = 'run'

    if creation_time is None:
        creation_time = datetime.datetime.now()

    dt = datetime.datetime.now().strftime('%Y.%m.%d-%H.%M')
    name = sanitize_filename(name)
    return root / f'{name}-{dt}-{generate_id()}'


def find_slurm_checkpoint(root: Path | str) -> Optional[Path]:
    root = Path(root)

    job_id = slurm_job_id()
    if job_id is None:
        return None

    for child in root.iterdir():
        if CheckpointDir(child).is_valid and CheckpointDir(child).slurm_job_id == job_id:
            return child

    return None


class CheckpointDir:
    def __init__(self, path: Path):
        self.path = Path(path).resolve()
        self.logger = logging.getLogger('dmlcloud')

    @property
    def config_file(self) -> Path:
        return self.path / 'config.yaml'

    @property
    def indicator_file(self) -> Path:
        return self.path / '.dmlcloud'

    @property
    def log_file(self) -> Path:
        return self.path / 'log.txt'

    @property
    def slurm_file(self) -> Path:
        return self.path / '.slurm-jobid'

    @property
    def exists(self) -> bool:
        return self.path.exists()

    @property
    def is_valid(self) -> bool:
        if not self.exists or not self.path.is_dir():
            return False

        if not self.indicator_file.exists():
            return False

        return True

    @property
    def slurm_job_id(self) -> Optional[str]:
        if not self.slurm_file.exists():
            return None

        with open(self.slurm_file) as f:
            return f.read()

    def create(self):
        if self.exists:
            raise ValueError(f'Checkpoint directory already exists: {self.path}')

        self.path.mkdir(parents=True, exist_ok=True)
        self.indicator_file.touch()
        self.log_file.touch()
        if slurm_job_id() is not None:
            with open(self.slurm_file, 'w') as f:
                f.write(slurm_job_id())

    def save_config(self, config: OmegaConf):
        if not self.exists:
            raise ValueError(f'Checkpoint directory does not exist: {self.path}')

        with open(self.config_file, 'w') as f:
            OmegaConf.save(config, f)

    def load_config(self) -> OmegaConf:
        if not self.is_valid:
            raise ValueError(f'Checkpoint directory is not valid: {self.path}')

        with open(self.config_file) as f:
            return OmegaConf.load(f)

    def __str__(self) -> str:
        return str(self.path)

    def __repr__(self) -> str:
        return f'CheckpointDir({self.path})'
