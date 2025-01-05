"""
Provides functions to interact with git
"""

import subprocess
import sys
import traceback
from pathlib import Path


def is_setuptools_cli_script(module):
    """
    Heuristically checks if the given module is a cli script generated by setuptools.
    """
    if not hasattr(module, '__file__'):
        return False
    try:
        with open(module.__file__) as f:
            lines = f.readlines(4089)
    except OSError:
        return False

    if len(lines) != 8:
        return False
    if not lines[0].startswith('#!'):
        return False
    if lines[2] != 'import re\n':
        return False
    if lines[3] != 'import sys\n':
        return False
    if lines[5] != "if __name__ == '__main__':\n":
        return False
    if 'sys.exit(' not in lines[7]:
        return False
    return True


def script_path():
    """
    Returns the path to the script or module that was executed.
    If python runs in interactive mode, or if "-c" command line option was used, raises a RuntimeError.
    """
    main = sys.modules['__main__']
    if not hasattr(main, '__file__'):
        raise RuntimeError('script_path() is not supported in interactive mode')

    if is_setuptools_cli_script(main):
        stack = traceback.extract_stack()
        if len(stack) < 2:
            return Path(main.__file__).resolve()
        return Path(stack[1].filename).resolve()

    else:
        return Path(main.__file__).resolve()


def script_dir():
    """
    Returns the directory containing the script or module that was executed.
    If python runs in interactive mode, or if "-c" command line option was used, then raises RuntimeError.
    """
    return script_path().parent


def project_dir():
    """
    Returns the top-level directory containing the script or module that was executed.
    If python runs in interactive mode, or if "-c" command line option was used, then raises RuntimeError.
    """
    cur_dir = script_dir()
    while (cur_dir / '__init__.py').exists():
        cur_dir = cur_dir.parent
    return cur_dir


def run_in_project(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **kwargs):
    """
    Runs a command in the project directory and returns the output.
    """
    cwd = project_dir()
    return subprocess.run(cmd, cwd=cwd, stdout=stdout, stderr=stderr, **kwargs)


def git_hash(short=False):
    if short:
        process = run_in_project(['git', 'rev-parse', '--short', 'HEAD'])
    else:
        process = run_in_project(['git', 'rev-parse', 'HEAD'])
    return process.stdout.decode('utf-8').strip()


def git_diff():
    """
    Returns the output of `git diff -U0 --no-color HEAD`
    """

    process = run_in_project(['git', 'diff', '-U0', '--no-color', 'HEAD'])
    return process.stdout.decode('utf-8').strip()
