"""
Provides functions to interact with git
"""

from .util.project import run_in_project


def git_hash(short=False):
    if short:
        process = run_in_project(['git', 'rev-parse', '--short', 'HEAD'])
    else:
        process = run_in_project(['git', 'rev-parse', 'HEAD'])
    return process.stdout.decode('utf-8').strip()


def git_diff():
    process = run_in_project(['git', 'diff', '-U0', '--no-color', 'HEAD'])
    return process.stdout.decode('utf-8').strip()
