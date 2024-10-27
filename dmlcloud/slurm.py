"""
Provides functions to interact with slurm
"""


import os


def slurm_job_id():
    """
    
    """
    return os.environ.get('SLURM_JOB_ID')


def slurm_step_id():
    return os.environ.get('SLURM_STEP_ID')


def slurm_available():
    return slurm_job_id() is not None
