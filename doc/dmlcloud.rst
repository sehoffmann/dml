.. currentmodule:: dmlcloud

dmlcloud
========

This the API reference for the dmlcloud package.

Core
----

.. autosummary::
    :toctree: generated

    Pipeline
    Stage


torch.distributed helpers
-------------------------
dmlcloud provides a set of helper functions to simplify the use of torch.distributed.

.. autosummary::
   :toctree: generated

    init

    is_root
    root_only
    root_first

    rank
    world_size
    local_rank
    local_world_size
    local_node

    all_gather_object
    gather_object
    broadcast_object

    has_slurm
    has_environment
    has_mpi


Logging
-------
dmlcloud provides a set of logging utilities to simplify logging in a distributed environment.
In particular, it lazily setups a logger ('dmlcloud') that only logs on the root process.
Users are encouraged to use the provided log functions instead of print statements to prevent duplicated logs.

.. autosummary::
   :toctree: generated

    logger
    log
    debug
    info
    warning
    error
    critical
    print_worker
    print_root
    setup_logger
    reset_logger




Metric Tracking
---------------
.. autosummary::
    :toctree: generated

    TrainingHistory
    Tracker
