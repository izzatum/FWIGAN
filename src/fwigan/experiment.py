"""Explicit experiment lifecycle. Epoch/batch always identify the next step.

A run owns immutable raw observations/config and mutable owned snapshots. Resumes
restore that run; warm starts create a new run and reset critic/optimizer/history.
Only trusted torch.save artifacts should be loaded.
"""
import copy
from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path
import platform
import time
import uuid
import deepwave
import numpy as np
import torch
from .config import ExperimentConfig
from .domain import finite_tensor, validate_physics, positive_int
from .parameters import InversionParameters
from .physics import LegacyDeepwave
from .data import save_observations, legacy_normalize, to_critic_layout
from .inversion import FWIStrategy, FWIGANStrategy
from .inversion.adversarial import _portable
from .checkpoints import (CHECKPOINT_SCHEMA, atomic_save, dataset_snapshot, restore_dataset,
    fingerprint_config, fingerprint_tensor, load_snapshot, rng_snapshot, restore_rng, validate_rng, seed_all)
from .evaluation import SolverCounts, CountedForward, evaluate_raw


@dataclass
class RunPosition:
    epoch: int = 0
    batch: int = 0
    global_step: int = 0


SNAPSHOT_FIELDS={'schema','kind','run_id','run_dir','config','config_fingerprint','dataset',
    'data_fingerprint','physical','strategy','observed_fingerprint','position','metrics','evaluations',
    'solver_counts','elapsed_seconds','rng'}


def _device(requested):
    if requested=='auto':return 'cuda' if torch.cuda.is_available() else 'cpu'
    if requested=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA requested but unavailable')
    if requested not in ('cpu','cuda'):raise ValueError('device must be cpu, cuda or auto')
    return requested


def _number_tree(value):
    if isinstance(value,dict):
        for item in value.values():_number_tree(item)
    elif isinstance(value,(tuple,list)):
        for item in value:_number_tree(item)
    elif isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
        raise ValueError('invalid nonfinite or nonnumeric metrics')


class Experiment:
    @classmethod
    def create(cls,config,data,output_root,forward=None):
        config=ExperimentConfig.from_dict(config.to_dict());config.device=_device(config.device)
        data=copy.deepcopy(data.to('cpu'));data.validate()
        seed_all(config.seed)
        run=cls._construct(config,data,forward)
        root=Path(output_root).resolve();root.mkdir(parents=True,exist_ok=True)
        run.run_id=uuid.uuid4().hex
        run.run_dir=root/f'{config.method}-{run.run_id}'
        run.run_dir.mkdir()  # Exclusive, so no existing artifacts can be replaced.
        run.resolved_config=config.to_dict()
        run.config_fingerprint=fingerprint_config(run.resolved_config)
        run.data_fingerprint=data.fingerprint()
        save_observations(run.run_dir/'observations.pt',data)
        (run.run_dir/'config.json').write_text(json.dumps(run.resolved_config,indent=2,allow_nan=False)+'\n')
        metadata=dict(run_id=run.run_id,config_fingerprint=run.config_fingerprint,data_fingerprint=run.data_fingerprint,
            python=platform.python_version(),platform=platform.platform(),torch=torch.__version__,
            numpy=np.__version__,deepwave=getattr(deepwave,'__version__','0.0.9'),device=config.device,
            cuda=torch.version.cuda)
        (run.run_dir/'run.json').write_text(json.dumps(metadata,indent=2)+'\n')
        return run

    @classmethod
    def _construct(cls,config,data,forward,physical=None,observed=None):
        run=cls();run.config=copy.deepcopy(config);run.data=data
        run._runtime_config_fingerprint=fingerprint_config(config.to_dict())
        run.position=RunPosition();run.metrics=[];run.evaluations=[]
        run.solver_counts=SolverCounts();run.elapsed_seconds=0.;run._saved_cuda_rng=[]
        run.forward=CountedForward(LegacyDeepwave() if forward is None else forward,run.solver_counts)
        device_data=data.to(config.device)
        source=device_data.source if config.method=='fwi' or device_data.initial_source is None else device_data.initial_source
        # Constructor requires a nonnegative initializer. Loading module state below
        # deliberately accepts finite signed learned legacy sigma.
        run.parameters=InversionParameters(device_data.initial_velocity,source,
            source.new_tensor(config.sigma_init),config.train_source,config.train_sigma).to(config.device)
        if physical is not None:
            cls._validate_physical(physical,run.parameters,device_data,config)
            run.parameters.load_state_dict(physical)
        if config.method=='fwigan':
            if observed is None:
                clean=to_critic_layout(legacy_normalize(device_data.traces))
                noisy=clean+config.observation_noise_level*torch.randn_like(clean)
                observed=(noisy if config.noise else clean)[0].permute(0,2,1)
            run.strategy=FWIGANStrategy(run.parameters,run.forward,device_data,config,observed=observed)
        else:run.strategy=FWIStrategy(run.parameters,run.forward,device_data,config)
        return run

    @staticmethod
    def _validate_physical(state,parameters,data,config):
        if not isinstance(state,dict) or set(state)!=set(parameters.state_dict()):raise ValueError('invalid physical state fields')
        for name,expected in parameters.state_dict().items():
            finite_tensor(state[name],name)
            if state[name].shape!=expected.shape:raise ValueError(f'incompatible physical {name} shape')
        validate_physics(state['velocity'],state['source'],data.acquisition.to('cpu'),data.grid,data.sampling)
        if config.method=='fwi':
            if not torch.equal(state['source'],data.source.cpu()):raise ValueError('fwi requires the fixed known source')
            if state['sigma'].item()!=state['sigma'].new_tensor(config.sigma_init).item():raise ValueError('fwi requires fixed sigma')
            if ((state['velocity']<config.velocity_min)|(state['velocity']>config.velocity_max)).any():
                raise ValueError('restored fwi velocity outside bounds')

    def _sync(self):
        if self.parameters.velocity.device.type=='cuda':torch.cuda.synchronize(self.parameters.velocity.device)

    def run(self,epochs=None,max_batches=None):
        """Run to total completed epochs. FWI budgets count whole logical epochs.

        max_batches bounds GAN steps, never a partial FWI accumulation. Returns
        self after checkpointing; full-data evaluation obeys its epoch cadence.
        """
        target=self.config.epochs if epochs is None else epochs
        positive_int(target,'epochs')
        target=min(target,self.config.epochs)
        if max_batches is not None:positive_int(max_batches,'max_batches')
        budget=0
        while self.position.epoch<target and (max_batches is None or budget<max_batches):
            self._sync();started=time.perf_counter()
            epoch,batch=self.position.epoch,self.position.batch
            if self.config.method=='fwigan':
                losses=self.strategy.step(slice(batch,None,self.config.batches))
                self.position.batch+=1
                finished=self.position.batch==self.config.batches
            else:
                losses=self.strategy.step_epoch();finished=True
            self._sync();elapsed=time.perf_counter()-started
            self.elapsed_seconds+=elapsed;self.position.global_step+=1;budget+=1
            self.metrics.append(dict(epoch=epoch,batch=batch,global_step=self.position.global_step,
                elapsed_seconds=elapsed,solver_forward=self.solver_counts.forward,solver_adjoint=self.solver_counts.adjoint,**losses))
            if finished:
                self.strategy.finish_epoch();self.position.epoch+=1;self.position.batch=0
                if self.position.epoch % self.config.evaluation_interval==0:self.evaluate()
                if self.position.epoch % self.config.checkpoint_interval==0:self.save_checkpoint()
        if self.position.epoch==self.config.epochs:
            # Reuse the cadence's deterministic prediction at this exact position.
            self.export_final()
        self.save_checkpoint()
        return self

    def evaluate(self):
        predicted,metrics=evaluate_raw(self.parameters,self.forward,self.strategy.data)
        row=dict(epoch=self.position.epoch,batch=self.position.batch,global_step=self.position.global_step,
            solver_forward=self.solver_counts.forward,solver_adjoint=self.solver_counts.adjoint,**metrics)
        self.evaluations.append(row)
        self._last_prediction=predicted;self._last_evaluation_step=self.position.global_step
        self._last_evaluation_physical=self._physical_fingerprint()
        return row.copy()

    def _physical_fingerprint(self):
        return tuple(fingerprint_tensor(value) for value in self.parameters.state_dict().values())

    def export_final(self,path=None):
        if (getattr(self,'_last_evaluation_step',None)!=self.position.global_step
                or getattr(self,'_last_evaluation_physical',None)!=self._physical_fingerprint()):self.evaluate()
        result=dict(schema=1,kind='result',run_id=self.run_id,data_fingerprint=self.data_fingerprint,
            config=self.resolved_config,position=asdict(self.position),predicted=self._last_prediction,
            metrics=self.evaluations[-1],counters=copy.deepcopy(self.strategy.counters),
            solver_counts=asdict(self.solver_counts),elapsed_seconds=self.elapsed_seconds,
            **_portable(self.parameters.state_dict()))
        return atomic_save(self.run_dir/'final.pt' if path is None else path,result)

    def save_checkpoint(self,path=None):
        if fingerprint_config(self.config.to_dict())!=self._runtime_config_fingerprint:
            raise ValueError('run configuration changed; use a new run')
        if (self.data.fingerprint()!=self.data_fingerprint
            or self.strategy.data.fingerprint()!=self.data_fingerprint):raise ValueError('immutable dataset fingerprint changed')
        observed=self.strategy.observed if self.config.method=='fwigan' else None
        payload=dict(schema=CHECKPOINT_SCHEMA,kind='checkpoint',run_id=self.run_id,run_dir=str(self.run_dir),
            config=copy.deepcopy(self.resolved_config),config_fingerprint=self.config_fingerprint,
            dataset=dataset_snapshot(self.data),data_fingerprint=self.data_fingerprint,
            physical=_portable(self.parameters.state_dict()),strategy=self.strategy.state_dict(),
            observed_fingerprint=None if observed is None else fingerprint_tensor(observed),
            position=asdict(self.position),metrics=copy.deepcopy(self.metrics),evaluations=copy.deepcopy(self.evaluations),
            solver_counts=asdict(self.solver_counts),elapsed_seconds=self.elapsed_seconds,rng=self._rng_snapshot())
        self._validate_structure(payload)
        self._validate_physical(payload['physical'],self.parameters,self.data,self.config)
        self.strategy._validate_state(payload['strategy'])
        return atomic_save(self.run_dir/'checkpoint.pt' if path is None else path,payload)

    def _rng_snapshot(self):
        state=rng_snapshot()
        # Retain states for devices absent from the current runtime.
        state['cuda'].extend(value.clone() for value in self._saved_cuda_rng[len(state['cuda']):])
        return state

    @staticmethod
    def _validate_structure(payload):
        if (set(payload)!=SNAPSHOT_FIELDS or type(payload.get('schema')) is not int
                or payload['schema']!=CHECKPOINT_SCHEMA or payload['kind']!='checkpoint'):
            raise ValueError('incompatible checkpoint schema or fields')
        if not isinstance(payload['run_id'],str) or not payload['run_id'] or not isinstance(payload['run_dir'],str):
            raise ValueError('invalid checkpoint run identity')
        cfg=payload['config']
        if not isinstance(cfg,dict) or set(cfg)!={f.name for f in fields(ExperimentConfig)}:raise ValueError('invalid resolved config fields')
        try:config=ExperimentConfig.from_dict(cfg)
        except (TypeError,OverflowError) as error:raise ValueError('invalid checkpoint configuration') from error
        if fingerprint_config(cfg)!=payload['config_fingerprint']:raise ValueError('config fingerprint mismatch')
        data=restore_dataset(payload['dataset'])
        if data.fingerprint()!=payload['data_fingerprint']:raise ValueError('data fingerprint mismatch')
        pos=payload['position']
        if not isinstance(pos,dict) or set(pos)!={'epoch','batch','global_step'}:raise ValueError('invalid next position')
        if any(type(v) is not int or v<0 for v in pos.values()):raise ValueError('invalid next position')
        batches=config.batches if config.method=='fwigan' else 1
        if (pos['epoch']>config.epochs or pos['batch']>=batches or
            (pos['epoch']==config.epochs and pos['batch']!=0) or pos['global_step']!=pos['epoch']*batches+pos['batch']):
            raise ValueError('inconsistent next epoch/batch position')
        if not isinstance(payload['metrics'],list) or len(payload['metrics'])!=pos['global_step']:
            raise ValueError('inconsistent metric history and position')
        metric_keys={'epoch','batch','global_step','elapsed_seconds','solver_forward','solver_adjoint','physical_updates','physical_skipped'}
        metric_keys|={'generator_loss','critic_loss','sigma','critic_updates','critic_skipped'} if config.method=='fwigan' else {'data_loss','regularization_loss','loss'}
        for index,row in enumerate(payload['metrics']):
            if not isinstance(row,dict) or set(row)!=metric_keys:raise ValueError('invalid metric fields')
            _number_tree(row)
            if row['epoch']!=index//batches or row['batch']!=index%batches or row['global_step']!=index+1:
                raise ValueError('inconsistent metric positions')
            if row['elapsed_seconds']<0:raise ValueError('invalid elapsed time')
            for key in ('epoch','batch','global_step','solver_forward','solver_adjoint'):
                if type(row[key]) is not int or row[key]<0:raise ValueError('invalid metric position or solver history')
            for key in ('physical_updates','physical_skipped'):
                if type(row[key]) is not int or row[key] not in (0,1):raise ValueError('invalid optimizer step metric')
            if row['physical_updates']+row['physical_skipped']!=1:raise ValueError('invalid physical update metrics')
            if config.method=='fwigan':
                for key in ('critic_updates','critic_skipped'):
                    if type(row[key]) is not int or row[key]<0:raise ValueError('invalid critic update metric')
                if row['critic_updates']+row['critic_skipped']!=config.critic_cycles:raise ValueError('invalid critic cycle metrics')
        if not isinstance(payload['evaluations'],list):raise ValueError('invalid evaluation history')
        for row in payload['evaluations']:
            keys={'epoch','batch','global_step','solver_forward','solver_adjoint','raw_residual_mse'}
            if data.reference_velocity is not None:keys.add('model_rmse')
            if not isinstance(row,dict) or set(row)!=keys:raise ValueError('invalid evaluation fields')
            _number_tree(row)
            for key in ('epoch','batch','global_step','solver_forward','solver_adjoint'):
                if type(row[key]) is not int or row[key]<0:raise ValueError('invalid evaluation position or solver history')
            if (row['raw_residual_mse']<0 or row.get('model_rmse',0)<0 or row['global_step']>pos['global_step']
                or row['epoch']>config.epochs or row['batch']>=batches
                or row['global_step']!=row['epoch']*batches+row['batch']):
                raise ValueError('invalid evaluation position or residual')
        counts=payload['solver_counts']
        if not isinstance(counts,dict) or set(counts)!={'forward','adjoint'} or any(type(v) is not int or v<0 for v in counts.values()):
            raise ValueError('invalid solver counters')
        for row in payload['metrics']+payload['evaluations']:
            if row['solver_forward']>counts['forward'] or row['solver_adjoint']>counts['adjoint']:
                raise ValueError('inconsistent solver counters and history')
        if counts['adjoint']>counts['forward']:raise ValueError('inconsistent solver/adjoint counters')
        _number_tree(payload['elapsed_seconds'])
        if payload['elapsed_seconds']<0:raise ValueError('invalid elapsed time')
        validate_rng(payload['rng'])
        strategy=payload['strategy']
        if not isinstance(strategy,dict):raise ValueError('invalid strategy state')
        if config.method=='fwigan':
            observed=strategy.get('observed');finite_tensor(observed,'observed',3)
            if observed.shape!=data.traces.shape or fingerprint_tensor(observed)!=payload['observed_fingerprint']:
                raise ValueError('observed target fingerprint mismatch')
        elif payload['observed_fingerprint'] is not None:raise ValueError('unexpected FWI target fingerprint')
        counters=strategy.get('counters')
        if not isinstance(counters,dict):raise ValueError('invalid strategy counters')
        for key,value in counters.items():
            if value!=sum(row.get(key,0) for row in payload['metrics']):raise ValueError('inconsistent strategy counters and history')
        for owner in ('physical','critic'):
            if owner=='critic' and config.method!='fwigan':continue
            scheduler=strategy.get(f'{owner}_scheduler')
            if not isinstance(scheduler,dict):raise ValueError('invalid scheduler fields')
            expected_epoch=0
            updates=0
            for index,row in enumerate(payload['metrics']):
                updates+=row.get(f'{owner}_updates',0)
                if (index+1)%batches==0 and updates:expected_epoch+=1
            if (scheduler.get('last_epoch')!=expected_epoch or scheduler.get('_step_count')!=expected_epoch+1
                or scheduler.get('step_size')!=getattr(config,f'{owner}_scheduler_step_size')
                or scheduler.get('gamma')!=config.scheduler_gamma):
                raise ValueError('inconsistent scheduler configuration or epoch state')
        return config,data

    @classmethod
    def resume(cls,checkpoint,device=None,forward=None):
        """Restore a run; move its directory with its immutable files for portability."""
        entry_rng=rng_snapshot()
        try:
            payload=load_snapshot(checkpoint);config,data=cls._validate_structure(payload)
            run_dir=Path(payload['run_dir'])
            # A copied directory is a valid relocation only with the same manifest.
            candidate=Path(checkpoint).resolve().parent
            if (candidate/'run.json').exists():run_dir=candidate
            cls._validate_artifacts(run_dir,payload)
            config.device=_device(config.device if device is None else device)
            observed=payload['strategy'].get('observed')
            run=cls._construct(config,data,forward,physical=payload['physical'],observed=observed)
            expected_keys=set(run.strategy.state_dict())
            if set(payload['strategy'])!=expected_keys:raise ValueError('invalid strategy fields')
            run.strategy.load_state_dict(payload['strategy'])
            # Optimizer.load_state_dict maps parameter moments to their device;
            # Adam's noncapturable step tensors must stay on CPU.
            run.run_id=payload['run_id'];run.run_dir=run_dir.resolve()
            run.resolved_config=payload['config'];run.config_fingerprint=payload['config_fingerprint']
            run.data_fingerprint=payload['data_fingerprint'];run.position=RunPosition(**payload['position'])
            run.metrics=payload['metrics'];run.evaluations=payload['evaluations'];run.elapsed_seconds=payload['elapsed_seconds']
            run.solver_counts.forward=payload['solver_counts']['forward'];run.solver_counts.adjoint=payload['solver_counts']['adjoint']
            run._saved_cuda_rng=[value.clone() for value in payload['rng']['cuda']]
            restore_rng(payload['rng'])  # Always last: initialization must not consume continuation RNG.
            return run
        except (KeyError,TypeError,RuntimeError,AttributeError,OSError) as error:
            restore_rng(entry_rng);raise ValueError('invalid checkpoint or immutable run artifacts') from error
        except Exception:
            restore_rng(entry_rng);raise

    @staticmethod
    def _validate_artifacts(run_dir,payload):
        try:
            manifest=json.loads((run_dir/'run.json').read_text())
            if (manifest['run_id']!=payload['run_id'] or manifest['config_fingerprint']!=payload['config_fingerprint']
                or manifest['data_fingerprint']!=payload['data_fingerprint']):raise ValueError('immutable run fingerprint mismatch')
            config=json.loads((run_dir/'config.json').read_text())
            if fingerprint_config(config)!=payload['config_fingerprint']:raise ValueError('immutable config fingerprint mismatch')
            data=restore_dataset(load_snapshot(run_dir/'observations.pt'))
            if data.fingerprint()!=payload['data_fingerprint']:raise ValueError('immutable observations fingerprint mismatch')
        except (KeyError,TypeError,OSError) as error:raise ValueError('missing or invalid immutable run artifacts') from error

    @classmethod
    def warm_start(cls,checkpoint,config,data,output_root,forward=None):
        """Transfer velocity, plus source/sigma for GAN; FWI keeps new known source.

        New config owns train/freeze choices; critic, optimizer, RNG, target and
        history all initialize anew. Grid/sampling must match the old experiment.
        """
        payload=load_snapshot(checkpoint);old_config,old_data=cls._validate_structure(payload)
        if old_data.grid!=data.grid or old_data.sampling!=data.sampling:
            raise ValueError('warm start requires matching grid and sampling geometry')
        initial=copy.deepcopy(data.to('cpu'));physical=payload['physical']
        # Validate against the old run before extracting any quantities.
        old_config.device='cpu'  # Validation needs no original CUDA runtime.
        old=cls._construct(old_config,old_data,forward,physical=physical,observed=payload['strategy'].get('observed'))
        old.strategy._validate_state(payload['strategy'])
        initial.initial_velocity=physical['velocity'].clone()
        if config.method=='fwigan':initial.initial_source=physical['source'].clone()
        initial.validate()
        run=cls.create(config,initial,output_root,forward)
        if config.method=='fwigan':
            with torch.no_grad():run.parameters.sigma.copy_(physical['sigma'].to(run.parameters.sigma.device))
        return run
