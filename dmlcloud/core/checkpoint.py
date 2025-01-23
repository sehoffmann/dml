import datetime
import secrets
from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf

from dmlcloud.slurm import slurm_job_id


__all__ = [
    'generate_checkpoint_path',
    'is_valid_checkpoint_dir',
    'create_checkpoint_dir',
    'find_slurm_checkpoint',
    'read_slurm_id',
    'save_config',
    'read_config',
]


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


def is_valid_checkpoint_dir(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False

    if not (path / '.dmlcloud').exists():
        return False

    return True


def create_checkpoint_dir(path: Path | str, name: Optional[str] = None) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / '.dmlcloud').touch()
    (path / 'log.txt').touch()
    if slurm_job_id() is not None:
        with open(path / '.slurm-jobid', 'w') as f:
            f.write(slurm_job_id())


def read_slurm_id(path: Path) -> Optional[str]:
    if is_valid_checkpoint_dir(path):
        return None

    if not (path / '.slurm-jobid').exists():
        return None

    with open(path / '.slurm-jobid') as f:
        return f.read()


def find_slurm_checkpoint(root: Path | str) -> Optional[Path]:
    root = Path(root)

    job_id = slurm_job_id()
    if job_id is None:
        return None

    for child in root.iterdir():
        if read_slurm_id(child) == job_id:
            return child

    return None


def save_config(config: OmegaConf, run_dir: Path):
    with open(run_dir / 'config.yaml', 'w') as f:
        OmegaConf.save(config, f)


def read_config(run_dir: Path) -> OmegaConf:
    with open(run_dir / 'config.yaml') as f:
        return OmegaConf.load(f)
