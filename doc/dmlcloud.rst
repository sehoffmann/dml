.. currentmodule:: dmlcloud

dmlcloud
========

This the API reference for the dmlcloud package.

.. autosummary::
    :toctree: generated

    TrainingPipeline
    Stage
    TrainValStage


Initialization
--------------
.. autosummary::
    :toctree: generated

    init_process_group_dummy
    init_process_group_slurm
    init_process_group_MPI
    init_process_group_auto


Metric Tracking
--------------
.. autosummary::
    :toctree: generated

    MetricReducer
    MetricTracker
    reduce_tensor
    Reduction


Distributed Helpers
-------------------
.. autosummary::
   :toctree: generated

    has_slurm
    has_environment
    has_mpi
    is_root
    root_only
    root_first
    rank
    world_size
    local_rank
    local_world_size
    local_node
    print_worker
    print_root
    all_gather_object
    gather_object
    broadcast_object