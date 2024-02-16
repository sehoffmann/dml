import logging
import random
import sys
from contextlib import nullcontext
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.distributed as dist
import wandb
from progress_table import ProgressTable
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ChainedScheduler, LinearLR

from dmlcloud.util import print_worker, project_dir, script_path, wandb_is_initialized, wandb_set_startup_timeout
from .checkpoint import resume_project_dir
from dmcloud.metrics import MetricTracker
from .scaling import scale_lr, scale_param_group
from .util import (
    git_hash,
    log_config,
    log_delimiter,
    log_diagnostics,
    log_git,
    log_model,
    setup_logging,
)




class BaseTrainer:

    def __init__(self, config):
        self.cfg = config

        self.initialized = False
        self.is_resumed = False
        self.tracker = MetricTracker()
        self.epoch = 1
        self.mode = 'train'


    def register_model(self, name, model, use_ddp=True):
        if use_ddp:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[self.device], broadcast_buffers=False
            )
        self.models[name] = model


    def register_optimizer(self, name, optimizer):
        self.optimizers[name] = optimizer


    @property
    def use_checkpointing(self):
        return self.model_dir is not None

    @property
    def is_gpu(self):
        return self.device.type == 'cuda'

    @property
    def model_dir(self):
        return self.cfg.model_dir

    @property
    def job_id(self):
        return self.cfg.job_id

    def setup_all(self, use_checkpointing=True, use_wandb=True, print_diagnostics=True):
        if self.initialized:
            raise ValueError('Trainer already initialized! Call reset() first.')

        if not dist.is_initialized():
            raise ValueError(
                'Default process group not initialized! Call torch.distributed.init_process_group() first.'
            )

        self.seed()
        self.setup_general()

        if use_checkpointing:
            self.is_resumed = resume_project_dir(self.cfg)

        if use_wandb:
            self.setup_wandb()

        if print_diagnostics:
            self.print_diagnositcs()

        self.setup_loss()
        self.setup_dataset()
        self.setup_model()
        self.setup_optimizer()
        self.resume_training()

        if print_diagnostics:
            log_config(self.cfg)

        self.setup_table()

        self.initialized = True

    def print_diagnositcs(self):
        log_delimiter()
        logging.info(f'Script path: {script_path()}')
        logging.info(f'Project dir: {project_dir()}')
        log_git()
        log_diagnostics(self.device)

    @root_only
    def setup_table(self):
        self.table = ProgressTable(columns=self.metric_names(), print_row_on_update=False)

    def setup_general(self):
        setup_logging()

        if torch.cuda.is_available():
            self.device = torch.device('cuda', hvd.local_rank())
        else:
            self.device = torch.device('cpu')

        torch.set_num_threads(8)
        self.cfg.git_hash = git_hash()

    def seed(self):
        if self.cfg.seed is None:
            seed = int.from_bytes(random.randbytes(4), byteorder='little')
            self.cfg.seed = dist.broadcast_object_list([seed])[0]

        np.random.seed(self.cfg.seed)
        random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        torch.cuda.manual_seed(self.cfg.seed)

    @root_only
    def setup_wandb(self):
        wandb_set_startup_timeout(600)
        exp_name = self.cfg.wb_name if self.cfg.wb_name else self.cfg.name
        wandb.init(
            project=self.cfg.wb_project,
            name=exp_name,
            tags=self.cfg.wb_tags,
            dir=self.model_dir,
            id=self.job_id,
            resume='must' if self.is_resumed else 'never',
            config=self.cfg.as_dictionary(),
        )

    def setup_dataset(self):
        logging.info('Creating dataset')
        dist.barrier()
        ts = datetime.now()
        if self.is_root:
            self.train_dl, self.val_dl = self.create_dataset()
            dist.barrier()
        else:
            dist.barrier()  # wait until rank 0 has created the dataset (e.g. downloaded it)
            self.train_dl, self.val_dl = self.create_dataset()
        logging.info(f'Dataset creation took {(datetime.now() - ts).total_seconds():.1f}s')

        if hasattr(self.train_dl, 'dataset') and hasattr(self.train_dl.dataset, '__len__'):
            train_samples = f'{len(self.train_dl.dataset)}'
        else:
            train_samples = 'N/A'
        train_sizes = hvd.allgather(torch.tensor([len(self.train_dl)]), name='train_dataset_size')
        train_sizes = [t.item() for t in train_sizes]
        msg = 'Train dataset:'
        msg += f'\n\t* Batches: {train_sizes[0]}'
        msg += f'\n\t* Batches (total): {sum(train_sizes)}'
        msg += f'\n\t* Samples (calculated): {sum(train_sizes) * self.cfg.batch_size}'
        msg += f'\n\t* Samples (raw): {train_samples}'
        logging.info(msg)
        if len(set(train_sizes)) > 1 and self.is_root:
            logging.warning(f'!!! Uneven train dataset batches: {train_sizes}')

        if self.val_dl is not None:
            if hasattr(self.val_dl, 'dataset') and hasattr(self.val_dl.dataset, '__len__'):
                val_samples = f'{len(self.val_dl.dataset)}'
            else:
                val_samples = 'N/A'

            val_sizes = hvd.allgather(torch.tensor([len(self.val_dl)]), name='val_dataset_size')
            val_sizes = [t.item() for t in val_sizes]
            msg = 'Train dataset:'
            msg += f'\n\t* Batches: {val_sizes[0]}'
            msg += f'\n\t* Batches (total): {sum(val_sizes)}'
            msg += f'\n\t* Samples (calculated): {sum(val_sizes) * self.cfg.batch_size}'
            msg += f'\n\t* Samples (raw): {val_samples}'
            logging.info(msg)
            if len(set(val_sizes)) > 1 and self.is_root:
                logging.warning(f'!!! Uneven val dataset batches: {val_sizes}')

        log_delimiter()

    def setup_model(self):
        logging.info('Creating model')

        model = self.create_model().to(self.device)
        self.model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.device], broadcast_buffers=False)
        log_model(self.model)
        if self.is_root and self.use_checkpointing:
            with open(self.model_dir / 'model.txt', 'w') as f:
                f.write(str(self.model))
        log_delimiter()

    def setup_loss(self):
        self.loss_fn = self.create_loss()

    def setup_optimizer(self):
        logging.info('Creating optimizer')
        optimizer = self.create_optimizer(self.model.parameters(), self.cfg.base_lr)
        lr_scale_factor = self.scale_optimizer(optimizer)
        self.optimizer = hvd.DistributedOptimizer(
            optimizer, named_parameters=self.model.named_parameters(), op=hvd.Adasum if self.cfg.adasum else hvd.Average
        )

        schedulers = []
        if self.cfg.rampup_epochs:
            linear_warmup = LinearLR(
                self.optimizer, start_factor=1 / lr_scale_factor, end_factor=1.0, total_iters=self.cfg.rampup_epochs
            )
            schedulers.append(linear_warmup)

        user_scheduler = self.create_scheduler()
        if isinstance(user_scheduler, list):
            schedulers.extend(user_scheduler)
        elif user_scheduler is not None:
            schedulers.append(user_scheduler)

        self.scheduler = ChainedScheduler(schedulers)
        self.scaler = GradScaler(enabled=self.cfg.mixed)

    def scale_optimizer(self, optimizer):
        use_gpu = self.device.type == 'cuda'
        _, lr_scale_factor = scale_lr(
            optimizer.defaults['lr'], self.cfg.batch_size, self.cfg.base_batch_size, self.cfg.adasum, use_gpu
        )
        logging.info(f'LR Scale Factor: {lr_scale_factor}')
        logging.info('Param-Groups:')
        for i, param_group in enumerate(optimizer.param_groups):
            param_group_cpy = dict(param_group)
            scaled_params = scale_param_group(param_group, self.cfg, use_gpu)
            scaled_params_cpy = dict(scaled_params)
            optimizer.param_groups[i] = scaled_params
            del param_group_cpy['params']
            del scaled_params_cpy['params']
            logging.info(f'[P{i}] Pre-scaled: {param_group_cpy}')
            logging.info(f'[P{i}] Scaled:     {scaled_params_cpy}')
        log_delimiter()
        return lr_scale_factor

    def load_state_dict(self, state_dict):
        self.epoch = state_dict['epoch']
        self.train_metrics = MetricSaver(state_dict['train_metrics'])
        self.val_metrics = MetricSaver(state_dict['val_metrics'])
        self.misc_metrics = MetricSaver(state_dict['misc_metrics'])
        self.model.load_state_dict(state_dict['model_state'])
        self.optimizer.load_state_dict(state_dict['optimizer_state'])
        self.scheduler.load_state_dict(state_dict['scheduler_state'])
        self.scaler.load_state_dict(state_dict['scaler_state'])

    def state_dict(self):
        state_dict = {
            'epoch': self.epoch,
            'train_metrics': self.train_metrics.epochs,
            'val_metrics': self.val_metrics.epochs,
            'misc_metrics': self.misc_metrics.epochs,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'scaler_state': self.scaler.state_dict(),
        }
        return state_dict

    def load_checkpoint(self, path):
        state_dict = torch.load(path, map_location=self.device)
        self.load_state_dict(state_dict)
        self.epoch += 1

    def resume_training(self):
        if not self.use_checkpointing:
            return

        cp_path = self.model_dir / 'checkpoint.pt'
        if cp_path.exists():
            self.load_checkpoint(cp_path)
            logging.info(f'Loaded checkpoint from {cp_path}')
            logging.info(
                f'Continuing training at epoch {self.epoch}, previous loss: {self.train_metrics.last["loss"]:.3f}'
            )
        elif self.is_resumed:
            logging.critical('No checkpoint found!')
            sys.exit(1)

    @root_only
    def save_checkpoint(self):
        if not self.use_checkpointing:
            return

        checkpoint_path = self.model_dir / 'checkpoint.pt'
        best_path = self.model_dir / 'best.pt'

        torch.save(self.state_dict(), checkpoint_path)
        if self.is_best_epoch():
            torch.save(self.state_dict(), best_path)
            if wandb_is_initialized():
                wandb.save(str(best_path), policy='now', base_path=str(self.model_dir))

        self.train_metrics.scalars_to_csv(self.model_dir / 'train_metrics.csv')
        self.val_metrics.scalars_to_csv(self.model_dir / 'val_metrics.csv')
        self.misc_metrics.scalars_to_csv(self.model_dir / 'misc_metrics.csv')

    def is_best_epoch(self):
        best_val_loss = min(self.val_metrics.get_metrics(self.val_loss_name))
        return self.val_metrics.last[self.val_loss_name] == best_val_loss

    def _update_table_entries(self):
        metric_names = self.metric_names()
        if isinstance(metric_names, list):
            metric_names = {name: name for name in metric_names}
        for display_name, metric_name in metric_names.items():
            splits = metric_name.split('/', 1)
            value = None
            if metric_name == 'Epoch':
                value = str(self.epoch)
            elif metric_name == 'ETA':
                value = str(self.misc_metrics.last['eta'])
            elif metric_name == 'ms/batch':
                value = f'{self.misc_metrics.last["ms_per_batch"]:.1f}'
            elif metric_name == 'time/epoch':
                value = str(self.misc_metrics.last['time_per_epoch'])
            elif len(splits) == 2:
                group, key = splits[0], splits[1]
                metrics = self.train_metrics if group == 'train' else self.val_metrics
                if key in metrics.last:
                    value = metrics.last[key]
            else:
                raise ValueError(f'Invalid metric name: {metric_name}')
            self.table[display_name] = value
        self.table.next_row()

    @root_only
    def log_epoch(self):
        self._update_table_entries()
        if wandb_is_initialized():
            self.log_wandb()

    def log_wandb(self):
        metrics = {}
        for key, value in self.train_metrics.scalar_metrics()[-1].items():
            metrics[f'train/{key}'] = value

        for key, value in self.val_metrics.scalar_metrics()[-1].items():
            metrics[f'val/{key}'] = value

        for key, value in self.misc_metrics.scalar_metrics()[-1].items():
            metrics[f'misc/{key}'] = value

        wandb.log(metrics)
        if self.is_best_epoch():
            wandb.run.summary['best/epoch'] = self.epoch
            for key, value in metrics.items():
                if not key.startswith('misc'):
                    wandb.run.summary[f'best/{key}'] = value

    def forward_step(self, batch_idx, batch):
        raise NotImplementedError()

    def switch_mode(self, train=True):
        if train:
            self.model.train()
            self.current_metrics = self.train_metrics
            self.mode = 'train'
        else:
            self.model.eval()
            self.current_metrics = self.val_metrics
            self.mode = 'eval'

    def pre_train(self):
        self.epoch_train_start = datetime.now()

    def post_train(self):
        self.epoch_train_end = datetime.now()

    def train_epoch(self, max_steps=None):
        self.pre_train()
        self.switch_mode(train=True)

        # Do this now, and not later, to immidiately show that a new epoch has started
        if self.is_root and 'Epoch' in self.table.columns:
            self.table['Epoch'] = self.epoch

        if hasattr(self.train_dl, 'sampler') and hasattr(self.train_dl.sampler, 'set_epoch'):
            self.train_dl.sampler.set_epoch(self.epoch)

        nan_ctx_manager = torch.autograd.detect_anomaly() if self.cfg.check_nans else nullcontext()
        for batch_idx, batch in enumerate(self.train_dl):
            if max_steps and batch_idx >= max_steps:
                break

            self.optimizer.zero_grad()

            with nan_ctx_manager:
                # forward pass
                with autocast(enabled=self.cfg.mixed):
                    loss = self.forward_step(batch_idx, batch)
                # backward pass
                self.scaler.scale(loss).backward()  # scale loss and, in turn, gradients to prevent underflow

            if loss.isnan() and not self.scaler.is_enabled():
                logging.critical(
                    'Got NaN loss but mixed precision training is disabled! This might be due to NaN values in the data or from diverging training.'
                )
                sys.exit(1)

            self.optimizer.synchronize()  # make sure all async allreduces are done
            self.scaler.unscale_(self.optimizer)  # now, unscale gradients again

            if self.cfg.log_gradients:
                norm = global_grad_norm(self.model.parameters())
                # self.log_metric('grad_norm', norm, allreduce=False, reduction='statistics')
                if self.cfg.clip_gradients:
                    self.log_metric(
                        'grad_norm/n_clipped', norm > self.cfg.clip_gradients, dist.ReduceOp.SUM, allreduce=False
                    )

            if self.cfg.clip_gradients:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_gradients)

            with self.optimizer.skip_synchronize():  # we already synchronized manually
                self.scaler.step(self.optimizer)
                self.scaler.update()  # adjust gradient scaling based on number of infs/nans

            if not torch.isnan(loss):  # mixed-precision might produce nan steps
                self.log_metric('loss', loss)
                self.misc_metrics.log_metric('n_nan', 0, dist.ReduceOp.SUM)
            else:
                self.misc_metrics.log_metric('n_nan', 1, dist.ReduceOp.SUM)

            self.misc_metrics.log_metric('n_steps', 1, dist.ReduceOp.SUM, allreduce=False)
            self.misc_metrics.log_metric('n_total_batches', 1, dist.ReduceOp.SUM, allreduce=True)

        self.misc_metrics.log_python_object('lr', self.scheduler.get_last_lr()[0])
        for k, v in self.scaler.state_dict().items():
            self.misc_metrics.log_python_object(f'scaler/{k}', v)

        if self.scheduler is not None:
            self.scheduler.step()

        self.post_train()

    def pre_eval(self):
        self.epoch_eval_start = datetime.now()

    def post_eval(self):
        self.epoch_eval_end = datetime.now()

        n_remaining = self.cfg.epochs - self.epoch
        per_epoch = (self.epoch_eval_end - self.start_time) / self.epoch
        per_epoch -= timedelta(microseconds=per_epoch.microseconds)
        eta = n_remaining * per_epoch
        self.misc_metrics.log_python_object('eta', str(eta))
        self.misc_metrics.log_python_object('time_per_epoch', str(per_epoch))

        n_train_batches = self.misc_metrics.current_metrics['n_total_batches'].reduce().item()
        per_step = (self.epoch_train_end - self.epoch_train_start) / n_train_batches
        per_step = per_step.total_seconds() * 1000
        self.misc_metrics.log_python_object('ms_per_batch', per_step)

        self.reduce_metrics()
        self.log_epoch()
        self.save_checkpoint()
        self.epoch += 1

    def reduce_metrics(self):
        self.train_metrics.reduce()
        self.val_metrics.reduce()
        self.misc_metrics.reduce()

    def evaluate_epoch(self, max_steps=None):
        self.pre_eval()
        self.switch_mode(train=False)

        if self.val_dl is not None:
            for batch_idx, batch in enumerate(self.val_dl):
                if max_steps and batch_idx >= max_steps:
                    break

                with torch.no_grad():
                    loss = self.forward_step(batch_idx, batch).item()
                    self.log_metric('loss', loss)

        self.post_eval()

    def pre_training(self):
        print_worker('READY')
        self.start_time = datetime.now()
        logging.info('Starting training...')

    def post_training(self):
        if self.is_root:
            self.table.close()
        logging.info('Training finished.')

    def train(self, max_steps=None, use_checkpointing=True, use_wandb=True, print_diagnostics=True):
        if not self.initialized:
            self.setup_all(
                use_checkpointing=use_checkpointing, use_wandb=use_wandb, print_diagnostics=print_diagnostics
            )

        self.pre_training()
        while self.epoch <= self.cfg.epochs:
            self.train_epoch(max_steps)
            self.evaluate_epoch(max_steps)
        self.post_training()

    def log_metric(self, name, value, reduction='avg', allreduce=True):
        self.current_metrics.log_metric(name, value, reduction, allreduce)

    def log_python_object(self, name, value):
        self.current_metrics.log_python_object(name, value, reduction=None, allreduce=False)

    def metric_names(self):
        """
        Returns a list or dictionary with custom metrics that are displayed during training.
        If a dictionary is returned, the keys are used as display names.
        """
        columns = ['Epoch', 'ETA', 'train/loss']
        if self.val_dl is not None:
            columns += [f'val/{self.val_loss_name}']
        columns += ['ms/batch', 'time/epoch']
        return columns
