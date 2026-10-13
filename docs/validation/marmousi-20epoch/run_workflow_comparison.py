"""Budgeted disposable whole-workflow comparison. Dormant unless --launch.

Reuses local REST authentication and owned-Pod cleanup; never embeds that client
or its credentials in the fixed GPU bootstrap. Original runners stay unchanged.
"""
from __future__ import annotations
import argparse
import base64
from contextlib import contextmanager
from decimal import Decimal, InvalidOperation, ROUND_FLOOR
import fcntl
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import secrets
import signal
import subprocess
import tarfile
import time
import uuid
import zlib

import run_package_test as package_runner
from model_asset_download import original_model_entry
from published_source_download import build_published_manifest, normalize_notebook, RAW_ROOT, PROJECT

ROOT=Path(__file__).resolve().parent
VALIDATION=Path('/workspace/fwigan-validation')
PRIOR=Decimal('0.53723588335249')
ARTIFACT_CAP=512*1024*1024
RAW_ARTIFACT_CAP=2*1024*1024*1024
SUPPORTED_GPUS=('NVIDIA A40','NVIDIA RTX A5000','NVIDIA GeForce RTX 3090','NVIDIA RTX A6000','NVIDIA L40')
BASELINE='c7ec47ed89ba1380ad7a02bbc4526b93675e76f3'


def positive(value):
    try:result=Decimal(str(value))
    except (InvalidOperation,ValueError,TypeError):raise ValueError('Invalid monetary rate') from None
    if not result.is_finite() or result<=0:raise ValueError('Rate must be positive and finite')
    return result


def guarded(rate):return positive(rate)*Decimal('1.10')+Decimal('.01')


class BudgetLedger:
    def __init__(self,path,budget=Decimal('10'),prior=PRIOR,contingency=Decimal('.50'),cleanup_seconds=600):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.config={}
        for key,value in [('budget',budget),('prior',prior),('contingency',contingency)]:
            try:number=Decimal(str(value))
            except (InvalidOperation,ValueError):raise ValueError('Invalid budget input') from None
            if not number.is_finite() or number<0:raise ValueError('Invalid budget input')
            self.config[key]=str(number)
        if Decimal(self.config['budget'])<=0 or cleanup_seconds!=600:raise ValueError('Invalid budget/deletion reserve')
        self.config['cleanup_seconds']=600
        with self.locked() as state:pass

    @contextmanager
    def locked(self):
        with self.path.with_suffix(self.path.suffix+'.lock').open('a') as lock:
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
            state=json.loads(self.path.read_text()) if self.path.exists() else dict(self.config,entries={})
            if any(state.get(key)!=value for key,value in self.config.items()):raise ValueError('Budget ledger configuration mismatch')
            yield state
            temp=self.path.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2,allow_nan=False));temp.replace(self.path)

    def available(self,state):
        spent=sum((Decimal(entry['actual_guarded_cost']) if entry.get('cleanup_verified') else
                   Decimal(entry['reserved_cost']) for entry in state['entries'].values()),Decimal('0'))
        return Decimal(state['budget'])-Decimal(state['prior'])-Decimal(state['contingency'])-spent

    def rate_ceiling(self,requested_seconds=14400):
        if type(requested_seconds) is not int or not 0<requested_seconds<=14400:raise ValueError('Work duration must be positive and <=4hours')
        with self.locked() as state:
            value=(self.available(state)*3600/Decimal(requested_seconds+600)-Decimal('.01'))/Decimal('1.10')
        result=value.quantize(Decimal('.01'),rounding=ROUND_FLOOR)
        if result<=0:raise ValueError('Budget cannot fund useful work')
        return result

    def plan(self,rate,requested_seconds=14400):
        rate=positive(rate)
        if type(requested_seconds) is not int or not 0<requested_seconds<=14400:raise ValueError('Work duration must be positive and <=4hours')
        with self.locked() as state:available=self.available(state)
        g=guarded(rate);affordable=int((available*3600/g).to_integral_value(rounding=ROUND_FLOOR))
        work=min(requested_seconds,affordable-600)
        if work<=0:raise ValueError('Budget cannot fund work plus cleanup')
        return {'rate_cap':str(rate),'guard_rate':str(g),'work_seconds':work,'cleanup_seconds':600,
                'total_seconds':work+600,'reserved_cost':str(g*Decimal(work+600)/3600)}

    def reserve(self,run_id,plan):
        with self.locked() as state:
            if run_id in state['entries'] or any(not row.get('cleanup_verified') for row in state['entries'].values()):
                raise RuntimeError('A workflow reservation is active or cleanup is unverified')
            rate=positive(plan['rate_cap']);work=plan['work_seconds']
            if type(work) is not int or not 0<work<=14400 or plan['cleanup_seconds']!=600 or plan['total_seconds']!=work+600:
                raise ValueError('Invalid lifecycle reservation')
            cost=guarded(rate)*Decimal(work+600)/3600
            if Decimal(plan['guard_rate'])!=guarded(rate) or Decimal(plan['reserved_cost'])!=cost or cost>self.available(state):
                raise ValueError('Reservation exceeds conservative budget')
            state['entries'][run_id]=dict(plan,phase='reserved_before_create',cleanup_verified=False,reserved_at=time.time())

    def complete(self,run_id,actual_rate,lifetime_seconds,cleanup_verified):
        rate=positive(actual_rate)
        duration=Decimal(str(lifetime_seconds))
        if not duration.is_finite() or duration<0:raise ValueError('Invalid Pod lifetime')
        with self.locked() as state:
            if run_id not in state['entries']:raise RuntimeError('Unknown run cannot release reservation')
            state['entries'][run_id].update(actual_rate=str(rate),lifetime_seconds=str(duration),
                actual_guarded_cost=str(guarded(rate)*duration/3600),cleanup_verified=bool(cleanup_verified),
                phase='complete' if cleanup_verified else 'cleanup_unverified',finished_at=time.time())


def select_candidate(catalogs,min_vram_gb=24,rate_ceiling=Decimal('1.94')):
    cap=positive(rate_ceiling);stock={}
    for dc in catalogs['stock']['dataCenters']:
        for gpu in dc['gpuAvailability']:
            status=gpu.get('stockStatus')
            if isinstance(status,str) and status and status.lower()!='none':stock.setdefault(gpu['gpuTypeId'],[]).append(dc['id'])
    rows={gpu['id']:gpu for gpu in catalogs['prices']['gpuTypes']}
    for identifier in SUPPORTED_GPUS:
        row=rows.get(identifier)
        if not row or not stock.get(identifier):continue
        try:rate=positive(row.get('securePrice'))
        except ValueError:continue
        memory=row.get('memoryInGb')
        if not isinstance(memory,(int,float)) or not math.isfinite(memory) or memory<min_vram_gb or rate>cap:continue
        regions=list(dict.fromkeys(stock[identifier]))
        return {'gpu_type_id':identifier,'memory_gb':memory,'rate':str(rate),'data_center_id':regions[0],'regions':regions}
    raise RuntimeError('No compatible >=24GB GPU with positive advisory stock inside budget rate ceiling')


def original_notebook_manifest(repo):
    files=[]
    for name in ('FWIGAN.ipynb','FWIGAN_Noisy.ipynb'):
        path='notebooks/'+name
        raw=subprocess.check_output(['git','-C',str(repo),'cat-file','blob',BASELINE+':'+path],timeout=20)
        payload=normalize_notebook(raw)
        files.append({'path':path,'url':RAW_ROOT+BASELINE+'/'+path,'raw_bytes':len(raw),
            'raw_sha256':hashlib.sha256(raw).hexdigest(),'payload_bytes':len(payload),'payload_sha256':hashlib.sha256(payload).hexdigest()})
    return {'project':PROJECT,'commit':BASELINE,'files':files}


BOOTSTRAP=r'''
import base64,hashlib,io,json,os,pathlib,secrets,signal,subprocess,tarfile,threading,time,traceback,zlib
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
cfg=json.loads(zlib.decompress(base64.b64decode('__CONFIG__')))
root=pathlib.Path('/tmp/fwigan-whole-workflow');root.mkdir(exist_ok=True)
repo=root/'repo';repo.mkdir(exist_ok=True)
original=root/'original';original.mkdir(exist_ok=True)
output=root/'workflow';output.mkdir(exist_ok=True)
nonce=cfg.pop('artifact_nonce');lock=threading.RLock()
state={'status':'starting','stage':'bootstrap','source_commit':cfg['source']['commit'],'epochs':20,'artifacts':None}
work_end=cfg['work_deadline_utc'];collect_end=cfg['collect_deadline_utc']
def update(**values):
    with lock:state.update(values)
def read_json(path,cap=4*1024*1024):
    if not path.is_file():return None
    if path.stat().st_size>cap:raise ValueError('Structured result exceeds declared size cap')
    value=json.loads(path.read_text());json.dumps(value,allow_nan=False);return value
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path=='/status':
            with lock:current=dict(state)
            try:current['progress']=read_json(output/'progress.json',64*1024)
            except Exception:current['progress_read_error']=True
            body=json.dumps(current,allow_nan=False).encode()
            self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Cache-Control','no-store')
            self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body);return
        if self.path!='/artifacts':self.send_error(404);return
        supplied=self.headers.get('X-FWIGAN-Artifact-Token','')
        if not isinstance(supplied,str) or len(supplied)>128 or not secrets.compare_digest(supplied,nonce):self.send_error(403);return
        with lock:metadata=state.get('artifacts')
        bundle=root/'artifacts.tar.gz'
        if not metadata or not bundle.is_file():self.send_error(404);return
        self.send_response(200);self.send_header('Content-Type','application/gzip');self.send_header('Content-Length',str(metadata['bytes']))
        self.send_header('X-FWIGAN-SHA256',metadata['sha256']);self.end_headers()
        with bundle.open('rb') as stream:
            while True:
                chunk=stream.read(1024*1024)
                if not chunk:break
                self.wfile.write(chunk)
    def log_message(self,*args):pass
class Server(ThreadingHTTPServer):
    def get_request(self):
        request,address=super().get_request();request.settimeout(20);return request,address
server=Server(('0.0.0.0',8080),Handler);threading.Thread(target=server.serve_forever,daemon=True).start()
def run(stage,command,limit,logname='setup.log'):
    update(stage=stage)
    remaining=min(limit,work_end-time.time())
    if remaining<=0:raise TimeoutError('Working budget deadline exhausted')
    with (root/logname).open('ab') as log:
        child=subprocess.Popen(command,cwd=repo,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:return child.wait(timeout=remaining)
        except BaseException:
            os.killpg(child.pid,signal.SIGKILL);child.wait();raise
        finally:
            tail=(root/logname).read_text(errors='replace')[-20000:]
            update(**{logname.replace('.','_')+'_tail':tail})
def required(stage,command,limit):
    if run(stage,command,limit):raise RuntimeError(stage+' failed')
def bundle_results():
    document=read_json(output/'artifact_manifest.json') or {'files':[],'omitted':[]}
    selected={row['path']:row for row in document.get('files',[])}
    for name in ('result.json','progress.json','artifact_manifest.json'):
        if (output/name).is_file():selected.setdefault(name,{})
    files=[];raw_total=0;allowed={'.json','.pt','.pth','.ipynb','.png','.svg','.npz','.npy','.csv','.txt','.log','.yaml','.yml','.md','.gz'}
    for name,declared in sorted(selected.items()):
        path=pathlib.PurePosixPath(name)
        if path.is_absolute() or not path.parts or any(part.startswith('.') or part in ('..','runtime','__pycache__') for part in path.parts):raise ValueError('Unsafe selected artifact path')
        target=output.joinpath(*path.parts)
        if target.suffix not in allowed or target.is_symlink() or not target.is_file() or not target.resolve().is_relative_to(output.resolve()):raise ValueError('Unsafe artifact type')
        if any(output.joinpath(*path.parts[:n]).is_symlink() for n in range(1,len(path.parts)+1)):raise ValueError('Symlinked artifact parent')
        size=target.stat().st_size;raw_total+=size
        if size>1024*1024*1024 or raw_total>2*1024*1024*1024:raise ValueError('Selected raw artifact cap exceeded')
        digest=hashlib.sha256()
        with target.open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''):
                if time.time()>=collect_end:raise TimeoutError('Artifact collection deadline exhausted')
                digest.update(chunk)
        sha=digest.hexdigest()
        if declared.get('bytes',size)!=size or declared.get('sha256',sha)!=sha:raise ValueError('Driver artifact manifest mismatch')
        files.append({'path':name,'bytes':size,'sha256':sha})
    manifest={'files':files,'omitted':document.get('omitted',[]),'source_commit':cfg['source']['commit'],'driver_sha256':cfg['driver_sha256']}
    body=json.dumps(manifest,sort_keys=True,allow_nan=False).encode();bundle=root/'artifacts.tar.gz'
    with tarfile.open(bundle,'w:gz') as archive:
        info=tarfile.TarInfo('bundle_manifest.json');info.size=len(body);archive.addfile(info,io.BytesIO(body))
        for entry in files:
            if time.time()>=collect_end:raise TimeoutError('Artifact packaging deadline exhausted')
            archive.add(output/entry['path'],arcname=entry['path'],recursive=False)
    size=bundle.stat().st_size
    if size>512*1024*1024:bundle.unlink();raise ValueError('Compressed artifact cap512MiB exceeded')
    digest=hashlib.sha256()
    with bundle.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return {'bytes':size,'sha256':digest.hexdigest(),'manifest':manifest}
try:
    assert tuple(__import__('sys').version_info[:2])==(3,10),'Python3.10 required'
    exec(compile(zlib.decompress(base64.b64decode(cfg['source_helper'])),'<verified-source-downloader>','exec'),globals())
    update(stage='verifying published current source and original notebooks')
    update(source_verification=download_published_source(cfg['source'],repo,time.monotonic()+max(0,work_end-time.time())))
    update(original_notebook_verification=download_published_source(cfg['original'],original,time.monotonic()+max(0,work_end-time.time())))
    exec(compile(zlib.decompress(base64.b64decode(cfg['model_helper'])),'<verified-original-model-downloader>','exec'),globals())
    update(model_verification=download_models(cfg['models'],repo,time.monotonic()+max(0,work_end-time.time())))
    driver=zlib.decompress(base64.b64decode(cfg['driver']));assert hashlib.sha256(driver).hexdigest()==cfg['driver_sha256']
    script=root/'notebook_workflow_comparison.py';script.write_bytes(driver)
    required('creating isolated Python3.10 venv',[__import__('sys').executable,'-m','venv',str(root/'venv')],60)
    py=str(root/'venv/bin/python')
    os.environ.update(MAX_JOBS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TORCH_CUDA_ARCH_LIST='8.6+PTX',
        MPLCONFIGDIR=str(root/'matplotlib'),XDG_CACHE_HOME=str(root/'cache'),MPLBACKEND='Agg',PYTHONUNBUFFERED='1')
    os.environ['PATH']=str(root/'venv/bin')+os.pathsep+os.environ['PATH']
    required('installing pinned CUDA and notebook stack',[py,'-m','pip','install','--no-cache-dir',
        '--extra-index-url','https://download.pytorch.org/whl/cu117','torch==1.13.1+cu117','numpy==1.23.5',
        'scipy==1.10.1','matplotlib==3.7.5','tqdm==4.67.1','PyYAML==6.0.2','pytest==8.4.2','setuptools==69.5.1',
        'wheel==0.45.1','ninja==1.13.0','nbformat==5.10.4','nbclient==0.10.2','ipykernel==6.31.0'],900)
    probe="import torch,json;assert torch.cuda.is_available();assert torch.cuda.device_count()==1;p=torch.cuda.get_device_properties(0);assert p.name=="+repr(cfg['expected_gpu'])+";assert (p.major,p.minor) in [(8,6),(8,9)];assert p.total_memory>=21*1024**3;print(json.dumps({'name':p.name,'bytes':p.total_memory,'capability':[p.major,p.minor],'torch':torch.__version__,'cuda':torch.version.cuda}))"
    if run('verifying actual GPU compatibility',[py,'-c',probe],60,'device.json'):raise RuntimeError('GPU compatibility probe failed')
    update(device=read_json(root/'device.json'))
    required('building DeepWave once',[py,'-m','pip','install','--no-cache-dir','--no-build-isolation','deepwave==0.0.9'],1200)
    required('installing exact package editable',[py,'-m','pip','install','--no-deps','--no-build-isolation','-e',str(repo)],120)
    code=run('running four CUDA integration checks',[py,'-m','pytest','tests/test_cuda.py','-ra','--junitxml='+str(root/'cuda.xml')],600,'cuda.log')
    import xml.etree.ElementTree as ET
    xml=ET.parse(root/'cuda.xml').getroot();suites=[xml] if xml.tag=='testsuite' else list(xml.iter('testsuite'))
    counts={key:sum(int(item.get(key,0)) for item in suites) for key in ('tests','failures','errors','skipped')}
    update(cuda_tests=counts)
    if code or counts!={'tests':4,'failures':0,'errors':0,'skipped':0}:raise RuntimeError('CUDA compatibility tests did not all pass')
    code=run('running whole old/new notebook workflows',[py,str(script),'--repo',str(repo),'--original-root',str(original),
        '--output-dir',str(output),'--epochs','20','--device','cuda','--deadline-utc',str(work_end-60),
        '--status-file',str(output/'progress.json')],max(1,work_end-time.time()),'workflow.log')
    update(workflow_exit_code=code,workflow_result=read_json(output/'result.json'))
    update(status='completed' if code==0 else 'failed')
except BaseException as error:
    update(status='partial' if isinstance(error,TimeoutError) else 'failed',error=type(error).__name__+': '+str(error),traceback=traceback.format_exc())
finally:
    try:update(workflow_result=read_json(output/'result.json'))
    except BaseException as error:update(result_collection_error=type(error).__name__)
    try:update(stage='packaging selected artifacts',artifacts=bundle_results())
    except BaseException as error:update(artifact_error=type(error).__name__+': '+str(error))
    update(stage='finished',finished_at=time.time())
while True:time.sleep(30)
'''


def make_bootstrap(config):
    source=BOOTSTRAP.replace('__CONFIG__',base64.b64encode(zlib.compress(json.dumps(config,allow_nan=False,separators=(',',':')).encode(),9)).decode())
    if 'RUNPOD_API_KEY' in source or 'GITHUB_TOKEN' in source:raise ValueError('Credential name may not reach Pod bootstrap')
    compile(source,'<whole workflow bootstrap>','exec')
    chunks=package_runner.bootstrap_chunks(source)
    if sum(map(len,chunks))>60*1024:raise ValueError('Fixed bootstrap exceeds60KiB argv cap')
    return source


def extract_artifacts(bundle,destination,expected):
    destination=Path(destination)
    if destination.exists():raise ValueError('Artifact destination must be new')
    destination.mkdir(parents=True)
    with tarfile.open(bundle,'r:gz') as archive:
        members=archive.getmembers();names={member.name for member in members}
        if len(names)!=len(members) or 'bundle_manifest.json' not in names:raise ValueError('Invalid artifact members')
        total=0
        for member in members:
            path=PurePosixPath(member.name)
            if not member.isfile() or path.is_absolute() or any(part.startswith('.') or part=='..' for part in path.parts):
                raise ValueError('Unsafe artifact archive member')
            total+=member.size
            if member.size>1024**3 or total>RAW_ARTIFACT_CAP:raise ValueError('Artifact expanded size cap exceeded')
        header=archive.getmember('bundle_manifest.json')
        if header.size>4*1024*1024:raise ValueError('Artifact manifest size cap exceeded')
        manifest=json.loads(archive.extractfile(header).read());json.dumps(manifest,allow_nan=False)
        if manifest!=expected:raise ValueError('Artifact manifest differs from status metadata')
        listed={entry['path']:entry for entry in manifest['files']}
        if len(listed)!=len(manifest['files']) or names!=set(listed)|{'bundle_manifest.json'}:raise ValueError('Artifact selection mismatch')
        for member in members:
            target=destination.joinpath(*PurePosixPath(member.name).parts)
            if not target.resolve().is_relative_to(destination.resolve()):raise ValueError('Unsafe artifact target')
            target.parent.mkdir(parents=True,exist_ok=True);digest=hashlib.sha256();size=0
            with archive.extractfile(member) as source,target.open('xb') as output:
                for chunk in iter(lambda:source.read(1024*1024),b''):digest.update(chunk);size+=len(chunk);output.write(chunk)
            if member.name in listed and (size!=listed[member.name]['bytes'] or digest.hexdigest()!=listed[member.name]['sha256']):
                raise ValueError('Artifact file checksum/length mismatch')
    return manifest


class WorkflowController(package_runner.Controller):
    def __init__(self,args,report,ledger,run_id,config,nonce):
        super().__init__(args,report)
        self.name='fwigan-whole-'+run_id;self.ledger=ledger;self.run_id=run_id
        self.config=config;self.nonce=nonce;self.deadline=None;self.collect_deadline=None
        self.actual_rate=None;self.artifact_attempted=False;self.watchdog=None
        self.watchdog_path=args.report.with_suffix('.watchdog.json')

    def catalog(self):
        queries={'stock':'query { dataCenters { id gpuAvailability { gpuTypeId stockStatus } } }',
                 'prices':'query { gpuTypes { id displayName memoryInGb securePrice communityPrice } }',
                 'templates':'query { podTemplates(input:{isRunpod:true}) { id name imageName isServerless isPublic isRunpod } }'}
        result={}
        for kind,query in queries.items():
            response=self.public.post('https://api.runpod.io/graphql',json={'query':query},timeout=(10,30),allow_redirects=False)
            response.raise_for_status();body=response.json()
            if body.get('errors'):raise RuntimeError('Public workflow catalog returned errors')
            result[kind]=body['data'];package_runner.save_report(body['data'],self.args.report.with_name(self.args.report.stem+'-'+kind+'.json'))
        return result

    def watchdog_state(self,finished=False):
        state={'name':self.name,'pod_id':self.created_id,'existing_ids':sorted(self.existing),
            'deadline_utc':self.report.get('cleanup_deadline_utc'),'finished':finished,
            'report_path':str(self.watchdog_path.with_suffix('.result.json'))}
        package_runner.save_report(state,self.watchdog_path);self.watchdog_path.chmod(0o600)

    def launch(self):
        self.existing={pod['id'] for pod in self.api('GET','/pods')};catalogs=self.catalog()
        cap=self.ledger.rate_ceiling(self.args.work_seconds);candidate=select_candidate(catalogs,rate_ceiling=cap)
        plan=self.ledger.plan(cap,self.args.work_seconds)
        template=next(row for row in catalogs['templates']['podTemplates'] if row['id']=='runpod-torch-v21')
        if not template['isRunpod'] or template['isServerless'] or 'py3.10' not in template['imageName']:raise ValueError('Unverified Python3.10 template')
        # Reserve the maximum allowed rate/lifetime, not the cheaper selected rate.
        self.ledger.reserve(self.run_id,plan);self.report.update(reservation=plan,candidate=candidate,
            existing_pod_ids=sorted(self.existing),catalog_rate_per_hour=float(candidate['rate']))
        t0=time.monotonic();utc=time.time();self.deadline=t0+plan['work_seconds'];self.collect_deadline=self.deadline+360
        self.config.update(expected_gpu=candidate['gpu_type_id'],work_deadline_utc=utc+plan['work_seconds'],collect_deadline_utc=utc+plan['work_seconds']+360)
        self.report.update(status='creating',created_pod_name=self.name,create_attempted=True,create_requested_at=utc,
            work_deadline_utc=self.config['work_deadline_utc'],cleanup_deadline_utc=utc+plan['total_seconds'])
        bootstrap=make_bootstrap(self.config)
        payload={'name':self.name,'cloudType':'SECURE','computeType':'GPU','gpuCount':1,
            'gpuTypeIds':[candidate['gpu_type_id']],'gpuTypePriority':'custom','dataCenterPriority':'availability',
            'imageName':template['imageName'],'containerDiskInGb':40,'volumeInGb':0,'interruptible':False,
            'minRAMPerGPU':32,'minVCPUPerGPU':4,
            'allowedCudaVersions':['11.8','12.0','12.1','12.2','12.3','12.4','12.5','12.6','12.7','12.8','12.9','13.0'],
            'ports':['8080/http'],'dockerEntrypoint':['python','-u','-c',
                'import base64,sys,zlib;exec(zlib.decompress(base64.b64decode("".join(sys.argv[1:]))))'],
            'dockerStartCmd':package_runner.bootstrap_chunks(bootstrap)}
        self.report['bootstrap_sha256']=hashlib.sha256(bootstrap.encode()).hexdigest()
        package_runner.save_report(self.report,self.args.report);self.watchdog_state()
        self.watchdog=subprocess.Popen([__import__('sys').executable,str(Path(__file__).resolve()),'--watchdog-state',str(self.watchdog_path)],
            stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
        pod=self.api('POST','/pods',json=payload);returned=pod['id']
        if returned in self.existing:raise RuntimeError('Create returned pre-existing Pod')
        readback=self.api('GET','/pods/'+returned+'?includeMachine=true')
        if readback.get('name')!=self.name:raise RuntimeError('Created Pod ownership did not verify')
        self.created_id=returned;self.watchdog_state()
        actual_gpu=(readback.get('machine') or {}).get('gpuTypeId')
        if actual_gpu!=candidate['gpu_type_id'] or readback.get('gpuCount')!=1:raise RuntimeError('GPU identity/count metadata missing or mismatched')
        rates=[positive(candidate['rate'])]
        for field in ('costPerHr','adjustedCostPerHr'):
            if readback.get(field) is not None:rates.append(positive(readback[field]))
        if len(rates)==1:raise RuntimeError('Created Pod billing rate missing')
        self.actual_rate=max(rates)
        self.report.update(pod_id=returned,status='running',reported_compute_rate_per_hour=float(self.actual_rate),
            created_pod={key:readback.get(key) for key in ('id','name','costPerHr','adjustedCostPerHr','gpuCount','imageName')})
        if self.actual_rate>cap:raise RuntimeError('Created Pod rate exceeds budget-derived ceiling')
        last_update=0
        while time.monotonic()<self.collect_deadline:
            now=time.monotonic()
            try:
                response=self.public.get(f'https://{returned}-8080.proxy.runpod.net/status',timeout=(5,15),allow_redirects=False)
                if response.status_code==200:
                    state=response.json();json.dumps(state,allow_nan=False);self.report['workflow_run']=state
                    package_runner.save_report(self.report,self.args.report)
                    if now-last_update>=30:
                        print('WORKFLOW_PROGRESS',json.dumps({'stage':state.get('stage'),'status':state.get('status'),
                            'progress':state.get('progress'),'elapsed_seconds':round(now-t0)}),flush=True);last_update=now
                    if state.get('stage')=='finished':
                        self.report['status']=state.get('status','failed');self.collect_artifacts(state);return
            except (self.requests.RequestException,ValueError):pass
            if now>=self.deadline and self.report.get('workflow_run',{}).get('stage')!='packaging selected artifacts':
                self.report['status']='partial_budget_deadline'
            time.sleep(min(5,max(0,self.collect_deadline-time.monotonic())))
        raise TimeoutError('Workflow/retrieval cutoff reached; deleting within reserved cleanup interval')

    def cleanup(self):
        state={'name':self.name,'pod_id':self.created_id,'existing_ids':sorted(self.existing),
               'deadline_utc':self.report.get('cleanup_deadline_utc') or time.time()+600}
        if not self.report.get('create_attempted'):
            self.report['cleanup_verified']=True;return
        owned_cleanup(state,self.auth,self.report,self.args.report)
        if self.report.get('create_requested_at'):
            self.report['pod_lifetime_seconds']=self.report['finished_at']-self.report['create_requested_at']
            self.report['estimated_compute_cost']=self.report['pod_lifetime_seconds']/3600*float(self.actual_rate or self.report['budget_rate_ceiling'])
        package_runner.save_report(self.report,self.args.report)

    def collect_artifacts(self,state=None):
        if self.artifact_attempted or self.created_id is None:return
        state=state or self.report.get('workflow_run') or {};metadata=state.get('artifacts')
        if not metadata:return
        self.artifact_attempted=True
        size=metadata.get('bytes');digest=metadata.get('sha256')
        if type(size) is not int or not 0<size<=ARTIFACT_CAP or not isinstance(digest,str) or not re.fullmatch('[0-9a-f]{64}',digest):
            raise ValueError('Artifact metadata exceeds fixed cap')
        target=self.args.report.with_suffix('.artifacts.tar.gz');temporary=target.with_suffix('.partial')
        if target.exists() or temporary.exists():raise ValueError('Refusing local artifact overwrite')
        deadline=self.collect_deadline or time.monotonic()+30;read=0;sha=hashlib.sha256()
        try:
            with self.public.get(f'https://{self.created_id}-8080.proxy.runpod.net/artifacts',
                headers={'X-FWIGAN-Artifact-Token':self.nonce},stream=True,timeout=(5,20),allow_redirects=False) as response:
                if response.status_code!=200 or response.headers.get('Content-Length')!=str(size) or response.headers.get('X-FWIGAN-SHA256')!=digest:
                    raise RuntimeError('Artifact response did not verify')
                with temporary.open('xb') as output:
                    for chunk in response.iter_content(1024*1024):
                        if time.monotonic()>=deadline:raise TimeoutError('Artifact retrieval reserve exhausted')
                        read+=len(chunk)
                        if read>size:raise ValueError('Artifact transfer exceeded expected length')
                        sha.update(chunk);output.write(chunk)
            if read!=size or sha.hexdigest()!=digest:raise ValueError('Artifact archive SHA/length mismatch')
            temporary.replace(target)
            manifest=extract_artifacts(target,self.args.report.with_suffix('.artifacts'),metadata['manifest'])
            self.report['artifact_retrieval']={'verified':True,'bundle':str(target),'sha256':digest,'bytes':read,'manifest':manifest}
        except self.requests.RequestException:
            raise RuntimeError('Artifact transfer failed before verification') from None
        finally:
            if temporary.exists():temporary.unlink()
            package_runner.save_report(self.report,self.args.report)


def owned_cleanup(state,session,report,target,sleep=time.sleep):
    """Keep retrying transient REST failures; never delete an unowned resource."""
    existing=set(state['existing_ids']);identifier=state.get('pod_id');empty_checks=0
    def request(method,path):
        response=session.request(method,package_runner.BASE+path,timeout=(3,8),allow_redirects=False)
        if response.status_code not in (200,202,204,404):response.raise_for_status()
        return response
    while True:
        try:
            if identifier is None:
                rows=request('GET','/pods').json()
                matches=[row for row in rows if row.get('name')==state['name'] and row['id'] not in existing]
                if len(matches)>1:raise ValueError('Ambiguous unique owned name')
                if not matches:
                    empty_checks+=1
                    if empty_checks>=3:
                        report.update(cleanup_verified=True,no_owned_pod_found=True,remaining_pod_ids=[row['id'] for row in rows],
                            cleanup_method='three_fresh_owned_name_absence_checks',finished_at=time.time())
                        package_runner.save_report(report,target);return
                    sleep(2);continue
                identifier=matches[0]['id'];empty_checks=0
            if identifier in existing:raise ValueError('Pre-existing resource refuses deletion')
            response=request('GET','/pods/'+identifier)
            if response.status_code==404:
                rows=request('GET','/pods').json()
                if all(row['id']!=identifier for row in rows):
                    report.update(pod_id=identifier,cleanup_verified=True,remaining_pod_ids=[row['id'] for row in rows],
                        cleanup_method='owned_get404_and_fresh_list_absence',finished_at=time.time())
                    package_runner.save_report(report,target);return
            else:
                row=response.json()
                if row.get('name')!=state['name']:
                    identifier=None;report['cleanup_ownership_reconciliation_required']=True;continue
                request('DELETE','/pods/'+identifier)
        except Exception as error:
            report.update(cleanup_verified=False,last_cleanup_error=type(error).__name__,
                cleanup_reserve_overdue=time.time()>state['deadline_utc'])
            package_runner.save_report(report,target)
        sleep(2)


def watchdog(path):
    state_path=Path(path)
    while True:
        state=json.loads(state_path.read_text())
        if state.get('finished'):return
        # Start420seconds after work cutoff, leaving180seconds of cleanup reserve.
        if time.time()>=state['deadline_utc']-180:break
        time.sleep(min(2,max(.1,state['deadline_utc']-180-time.time())))
    args=argparse.Namespace(report=Path(state['report_path']))
    report={'create_attempted':True,'status':'watchdog_cleanup'}
    client=package_runner.Controller(args,report)
    owned_cleanup(state,client.auth,report,args.report)


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=Path('/workspace/FWIGAN'))
    parser.add_argument('--driver',type=Path,default=ROOT/'notebook_workflow_comparison.py')
    parser.add_argument('--report',type=Path,default=VALIDATION/('whole-workflow-'+time.strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:6]+'.json'))
    parser.add_argument('--ledger',type=Path,default=VALIDATION/'whole-workflow-budget-ledger.json')
    parser.add_argument('--work-seconds',type=int,default=14400)
    parser.add_argument('--epochs',type=int,choices=[20],default=20)
    parser.add_argument('--launch',action='store_true')
    parser.add_argument('--watchdog-state',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    if not 1200<=args.work_seconds<=14400:parser.error('Work clock must be1200..14400seconds')
    return args


def prepare(args):
    if args.report.exists():raise ValueError('Fresh unique workflow report path required')
    archive,identity=package_runner.package(args.repo,omit_documentation_assets=True)
    if identity['missing']:raise ValueError('Required product files absent')
    source=build_published_manifest(args.repo,identity)
    driver=args.driver.read_bytes()
    if len(driver)>256*1024 or package_runner.SECRET.search(driver) or b'RUNPOD_API_KEY' in driver or b'GITHUB_TOKEN' in driver:
        raise ValueError('External driver exceeds cap or contains a prohibited credential name/value')
    config={'source':source,'original':original_notebook_manifest(args.repo),
        'models':[original_model_entry(args.repo,name) for name in ('marmousi_3.bin','marmousi_30.bin')],
        'source_helper':base64.b64encode(zlib.compress((ROOT/'published_source_download.py').read_bytes(),9)).decode(),
        'model_helper':base64.b64encode(zlib.compress((ROOT/'model_asset_download.py').read_bytes(),9)).decode(),
        'driver':base64.b64encode(zlib.compress(driver,9)).decode(),'driver_sha256':hashlib.sha256(driver).hexdigest(),
        'artifact_nonce':secrets.token_urlsafe(32),'expected_gpu':SUPPORTED_GPUS[0],'work_deadline_utc':time.time()+args.work_seconds,
        'collect_deadline_utc':time.time()+args.work_seconds+480}
    bootstrap=make_bootstrap(config);ledger=BudgetLedger(args.ledger);cap=ledger.rate_ceiling(args.work_seconds)
    report={'status':'prepared','launch_requested':args.launch,'pod_id':None,'cleanup_verified':False,
        'source_commit':identity['commit'],'published_source':source,'original_notebooks':config['original'],
        'models':config['models'],'driver_sha256':config['driver_sha256'],'epochs':20,
        'source_manifest_sha256':hashlib.sha256(json.dumps(source,sort_keys=True).encode()).hexdigest(),
        'budget_rate_ceiling':str(cap),'budget_plan':ledger.plan(cap,args.work_seconds),
        'bootstrap_sha256':hashlib.sha256(bootstrap.encode()).hexdigest(),
        'bootstrap_command_bytes':sum(map(len,package_runner.bootstrap_chunks(bootstrap))),
        'workflow_kind':'whole_notebook_quality_comparison_not_accepted_benchmark_performance'}
    return report,config,ledger


def main(argv=None):
    args=parse_args(argv)
    if args.watchdog_state:return watchdog(args.watchdog_state)
    report,config,ledger=prepare(args);package_runner.save_report(report,args.report)
    print('WORKFLOW_PREPARED',json.dumps({'report':str(args.report),'commit':report['source_commit'],
        'driver_sha256':report['driver_sha256'],'rate_ceiling':report['budget_rate_ceiling'],
        'bootstrap_bytes':report['bootstrap_command_bytes'],'epochs':20,'launch':args.launch}),flush=True)
    if not args.launch:return 0
    run_id=uuid.uuid4().hex[:12];controller=WorkflowController(args,report,ledger,run_id,config,config['artifact_nonce'])
    def stop(signum,frame):raise SystemExit('Interrupted; ownership cleanup required')
    old={sig:signal.signal(sig,stop) for sig in (signal.SIGINT,signal.SIGTERM)}
    try:controller.launch()
    except BaseException as error:
        report.update(status='failed',error=type(error).__name__+': '+str(error));package_runner.save_report(report,args.report)
        raise
    finally:
        for sig in old:signal.signal(sig,signal.SIG_IGN)
        try:
            try:controller.collect_artifacts()
            except BaseException as error:report['artifact_collection_error']=type(error).__name__
            try:controller.cleanup()
            finally:
                if run_id in json.loads(args.ledger.read_text())['entries']:
                    end=time.time();duration=max(0,end-report.get('create_requested_at',end))
                    ledger.complete(run_id,controller.actual_rate or positive(report['budget_rate_ceiling']),duration,report.get('cleanup_verified',False))
                controller.watchdog_state(finished=report.get('cleanup_verified',False))
        finally:
            package_runner.save_report(report,args.report)
            for sig,handler in old.items():signal.signal(sig,handler)
    return 0


if __name__=='__main__':raise SystemExit(main())
