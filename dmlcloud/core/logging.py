"""
Provides a simple logging interface for dmlcloud.

The dmlcloud logger is setup to only log messages on the root process, with severity 'INFO' or higher.
Non-root processes will only log messages with severity 'WARNING' or higher.

Attributes:
    logger (logging.Logger): The dmlcloud logger. Only logs messages on the root process.
"""

import logging
import sys
import warnings

import torch
import torch.distributed

from . import distributed as dmldist


logger = logging.getLogger('dmlcloud')


__all__ = [
    'logger',
    'log',
    'debug',
    'info',
    'warning',
    'error',
    'critical',
    'setup_logger',
    'reset_logger',
    'print_worker',
    'print_root',
]


def setup_logger():
    """
    Setup the dmlcloud logger.

    For this, torch.distributed must be initialized, as the logger will be setup to only log on the root process.
    Non-root processes will only log messages with severity 'WARNING' or higher.

    Usually, this function is called automatically when logging a message, and should not be called manually.
    """
    if logger.hasHandlers():
        warnings.warn('Logger already setup. Ignoring call to setup_logger().')
        return

    if not torch.distributed.is_initialized():
        warnings.warn(
            'Cannot setup logger without torch.distributed being initialized. Ignoring call to setup_logger(). Messages will be logged by all ranks!'
        )
        return

    logger.setLevel(logging.INFO if torch.distributed.get_rank() == 0 else logging.WARNING)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.addFilter(lambda record: record.levelno < logging.WARNING)
    stdout_handler.setFormatter(logging.Formatter())
    logger.addHandler(stdout_handler)

    stderr_handler = logging.StreamHandler()
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(logging.Formatter())
    logger.addHandler(stderr_handler)


def reset_logger():
    """
    Reset the dmlcloud logger to its initial state.

    This will remove all handlers from the logger and set its level to NOTSET.
    """
    logger.setLevel(logging.NOTSET)
    to_remove = list(logger.handlers)
    for handler in to_remove:
        logger.removeHandler(handler)


def log(level, msg, *args, exc_info=None, stack_info=False, extra=None):
    """
    Log 'msg % args' with severity 'level' on the dmlcloud logger.

    If the dmlcloud logger was not already setup, this function will setup the logger with the default configuration.
    """
    if not logger.hasHandlers():
        setup_logger()

    logger.log(level, msg, *args, exc_info=exc_info, stack_info=stack_info, extra=extra)


def debug(msg, *args, exc_info=None, stack_info=False, extra=None):
    """
    Log 'msg % args' with severity 'TRACE' on the dmlcloud logger.

    If the dmlcloud logger was not already setup, this function will setup the logger with the default configuration.
    """
    log(logging.DEBUG, msg, *args, exc_info=exc_info, stack_info=stack_info, extra=extra)


def info(msg, *args, exc_info=None, stack_info=False, extra=None):
    """
    Log 'msg % args' with severity 'INFO' on the dmlcloud logger.

    If the dmlcloud logger was not already setup, this function will setup the logger with the default configuration.
    """
    log(logging.INFO, msg, *args, exc_info=exc_info, stack_info=stack_info, extra=extra)


def warning(msg, *args, exc_info=None, stack_info=False, extra=None):
    """
    Log 'msg % args' with severity 'WARNING' on the dmlcloud logger.

    If the dmlcloud logger was not already setup, this function will setup the logger with the default configuration.
    """
    log(logging.WARNING, msg, *args, exc_info=exc_info, stack_info=stack_info, extra=extra)


def error(msg, *args, exc_info=None, stack_info=False, extra=None):
    """
    Log 'msg % args' with severity 'ERROR' on the dmlcloud logger.

    If the dmlcloud logger was not already setup, this function will setup the logger with the default configuration.
    """
    log(logging.ERROR, msg, *args, exc_info=exc_info, stack_info=stack_info, extra=extra)


def critical(msg, *args, exc_info=None, stack_info=False, extra=None):
    """
    Log 'msg % args' with severity 'CRITICAL' on the dmlcloud logger.

    If the dmlcloud logger was not already setup, this function will setup the logger with the default configuration.
    """
    log(logging.CRITICAL, msg, *args, exc_info=exc_info, stack_info=stack_info, extra=extra)


def print_worker(*values, sep=' ', end="\n", file=None, flush=True, barrier=False):
    """
    Print the values to a stream, default sys.stdout, with additional information about the worker.

    Args:
        values (Any): The values to print.
        sep (str, optional): The separator between arguments. Default is a space.
        end (str, optional): The string to append at the end of the message. Default is a newline.
        file (file, optional): The file to write the message to. Default is None.
        flush (bool, optional): If True, the output buffer is flushed. Default is True.
        barrier (bool, optional): If True, a barrier is inserted before and after printing. Default is False.
    """

    if barrier:
        torch.distributed.barrier()
    modified_values = [f'Worker {dmldist.rank()}']
    if dmldist.local_node() is not None:
        modified_values += [f'({dmldist.local_node()}.{dmldist.local_rank()})']
    modified_values.extend(values)
    print(*modified_values, sep=sep, end=end, file=file, flush=flush)
    if barrier:
        torch.distributed.barrier()


@dmldist.root_only
def print_root(*values, sep=' ', end="\n", file=None, flush=True):
    """
    Print the values to a stream if the current rank is the root rank.

    Default is to print to the standard output stream.

    Args:
        msg (str): The message to print.
        sep (str, optional): The separator between arguments. Default is a space.
        end (str, optional): The string to append at the end of the message. Default is a newline.
        file (file, optional): The file to write the message to. Default is None.
        flush (bool, optional): If True, the output buffer is flushed. Default is True.
    """

    print(*values, sep=sep, end=end, file=file, flush=flush)


if __name__ == '__main__':
    from .distributed import init

    init()

    debug('This is a debug message')
    info('This is an info message')
    warning('This is a warning message')
    error('This is an error message')
    critical('This is a critical message')
    reset_logger()
    torch.distributed.destroy_process_group()

    debug('This is a debug message')
    info('This is an info message')
    warning('This is a warning message')
    error('This is an error message')
    critical('This is a critical message')
