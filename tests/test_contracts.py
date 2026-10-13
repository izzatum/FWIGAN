import copy
import hashlib
import importlib
import json
from pathlib import Path

import deepwave as dw
import pytest
import torch

from reference import FROZEN, Generator, data_normalization, add_noise


def package():
    assert importlib.util.find_spec('fwigan') is not None, 'numerical package is missing'
    return importlib.import_module('fwigan')


def specs():
    f = package()
    grid = f.GridSpec((40, 70), 30.)
    sampling = f.SamplingSpec(64, .003)
    xs = torch.tensor([[[0., 300.]], [[0., 900.]]])
    xr = torch.zeros(2, 4, 2)
    xr[..., 1] = torch.arange(4) * 300.
    acq = f.Acquisition(xs, xr)
    source = dw.wavelets.ricker(25., sampling.nt, sampling.dt, .04).reshape(1, 1, -1)
    return f, grid, sampling, acq, source


def observations():
    f, grid, sampling, acq, source = specs()
    return f.ObservationData(torch.zeros(2, 4, 64), acq, grid, sampling, source,
                             torch.full(grid.shape, 2000.))


def test_frozen_reference_provenance():
    assert FROZEN['revision'] == 'c7ec47ed89ba1380ad7a02bbc4526b93675e76f3'
    for cells in FROZEN['notebooks'].values():
        for cell in cells.values():
            assert hashlib.sha256(cell['source'].encode()).hexdigest() == cell['sha256']


def test_package_exists():
    package()


def test_reference_traces_normalized_loss_and_gradients():
    f, grid, sampling, acq, source = specs()
    from fwigan.physics import LegacyDeepwave
    from fwigan.data.preprocessing import legacy_normalize, to_critic_layout, from_critic_layout
    velocity = torch.full(grid.shape, 2000., requires_grad=True)
    source = source.clone().requires_grad_()
    actual = LegacyDeepwave().predict(velocity, source, acq, grid, sampling)
    reference_velocity = velocity.detach().clone().requires_grad_()
    reference_source = source.detach().permute(2, 0, 1).clone().requires_grad_()
    generator = Generator(reference_velocity, torch.tensor(1.), reference_source, 2, 30., .003)
    expected = generator(acq.sources, acq.receivers)
    normalized = to_critic_layout(legacy_normalize(actual))
    torch.testing.assert_close(normalized, expected, rtol=2e-5, atol=2e-6)
    loss = normalized.square().mean()
    expected_loss = expected.square().mean()
    loss.backward(); expected_loss.backward()
    torch.testing.assert_close(loss, expected_loss, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(velocity.grad, reference_velocity.grad, rtol=1e-4, atol=1e-7)
    torch.testing.assert_close(source.grad.permute(2,0,1), reference_source.grad, rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(from_critic_layout(to_critic_layout(actual)), actual)
    direct = dw.scalar.Propagator({'vp': velocity.detach()}, 30.)(
        source.detach().permute(2,0,1).repeat(1,2,1), acq.sources, acq.receivers, .003)
    torch.testing.assert_close(actual, direct.permute(1,2,0), rtol=2e-5, atol=2e-6)
    doubled = LegacyDeepwave().predict(velocity.detach(), source.detach()*2, acq, grid, sampling)
    torch.testing.assert_close(doubled, actual.detach()*2, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(LegacyDeepwave().predict(velocity.detach(), source.detach(), acq, grid, sampling), actual.detach(), rtol=0, atol=0)


def test_legacy_normalization_uses_abs_of_max_and_preserves_noise_draw_order():
    package()
    from fwigan.data.preprocessing import legacy_normalize, to_critic_layout
    traces = torch.tensor([[[-4., -2., -1.], [-3., 1., 2.]]])
    expected = data_normalization(traces.permute(2,0,1)).permute(1,2,0)
    torch.testing.assert_close(legacy_normalize(traces), expected)
    assert legacy_normalize(traces)[0,0,0] == -4.
    torch.manual_seed(42)
    ref, _ = add_noise(to_critic_layout(expected), .05)
    torch.manual_seed(42)
    actual = to_critic_layout(legacy_normalize(traces))
    actual = actual + .05 * torch.randn_like(actual)
    torch.testing.assert_close(actual, ref, rtol=0, atol=0)


def test_origin_and_shot_selection():
    f, grid, sampling, acq, source = specs()
    from fwigan.physics import LegacyDeepwave
    origin = (60.,120.)
    shifted = f.Acquisition(acq.sources + torch.tensor(origin), acq.receivers + torch.tensor(origin))
    velocity = torch.full(grid.shape, 2000.)
    torch.testing.assert_close(LegacyDeepwave().predict(velocity,source,acq,grid,sampling),
        LegacyDeepwave().predict(velocity,source,shifted,f.GridSpec(grid.shape,30.,origin),sampling))
    assert acq.select([1]).sources.shape == (1,1,2)
    assert acq.select(slice(0,2,2)).receivers.shape == (1,4,2)
    assert acq.to('cpu').sources.device.type == 'cpu'


@pytest.mark.parametrize('field,value', [('shape',(0,70)),('shape',(40,)),('spacing',0.),('origin',(float('nan'),0.)),('pml_width',-1),('accuracy',8)])
def test_invalid_grid(field,value):
    f=package();kwargs={'shape':(40,70),'spacing':30.};kwargs[field]=value
    with pytest.raises(ValueError): f.GridSpec(**kwargs)


@pytest.mark.parametrize('nt,dt', [(0,.003),(64,0.),(64,float('nan')),(1.5,.003)])
def test_invalid_sampling(nt,dt):
    f=package()
    with pytest.raises(ValueError): f.SamplingSpec(nt,dt)


@pytest.mark.parametrize('mutation', ['double','nonfinite','negative_velocity','wrong_traces','outside','off_grid','wrong_source'])
def test_invalid_observations(mutation):
    data=observations();args=dict(traces=data.traces,acquisition=data.acquisition,grid=data.grid,
        sampling=data.sampling,source=data.source,initial_velocity=data.initial_velocity)
    if mutation=='double':args['traces']=data.traces.double()
    if mutation=='nonfinite':args['traces']=torch.full_like(data.traces,float('nan'))
    if mutation=='negative_velocity':args['initial_velocity']=-data.initial_velocity
    if mutation=='wrong_traces':args['traces']=data.traces[:1]
    if mutation in ('outside','off_grid'):
        sources=data.acquisition.sources.clone();sources[0,0,1]=3000. if mutation=='outside' else 301.
        args['acquisition']=package().Acquisition(sources,data.acquisition.receivers)
    if mutation=='wrong_source':args['source']=data.source[...,:20]
    with pytest.raises(ValueError):package().ObservationData(**args)


def test_forward_rejects_invalid_velocity_and_source():
    f,grid,sampling,acq,source=specs()
    from fwigan.physics import LegacyDeepwave
    for velocity,src in [(torch.full(grid.shape,-1.),source),(torch.ones(2,2),source),
                          (torch.ones(grid.shape),source.double()),(torch.ones(grid.shape),source[...,:2])]:
        with pytest.raises(ValueError):LegacyDeepwave().predict(velocity,src,acq,grid,sampling)


def test_parameters_register_frozen_state_and_named_groups():
    f,grid,sampling,acq,source=specs()
    from fwigan.parameters import InversionParameters
    from fwigan.config import ExperimentConfig
    config=ExperimentConfig(noise=True)
    params=InversionParameters(torch.full(grid.shape,2000.),source,torch.tensor(1.),train_sigma=True).to('cpu')
    assert set(params.state_dict()) == {'velocity','source','sigma'}
    groups=params.optimizer_groups(config)
    assert [(g['name'],g['lr']) for g in groups] == [('velocity',5.),('source',.001),('sigma',.01)]
    optimizer=torch.optim.Adam(groups)
    assert all(g['betas']==(.5,.9) for g in optimizer.param_groups)
    frozen=InversionParameters(params.velocity,source,torch.tensor(1.),train_source=False)
    assert set(dict(frozen.named_buffers())) == {'source','sigma'}
    assert [g['name'] for g in frozen.optimizer_groups(ExperimentConfig(method='fwi'))] == ['velocity']
    restored=InversionParameters(torch.ones_like(params.velocity),source,torch.tensor(0.),train_sigma=True)
    restored.load_state_dict(params.state_dict())
    torch.testing.assert_close(restored.sigma,params.sigma)


def test_config_roundtrip_and_legacy_defaults():
    package()
    from fwigan.config import ExperimentConfig
    c=ExperimentConfig()
    assert ExperimentConfig.from_dict(json.loads(json.dumps(c.to_dict())))==c
    assert (c.velocity_lr,c.source_lr,c.sigma_lr,c.critic_lr)==(5.,.001,.01,.001)
    assert (c.physical_scheduler_step_size,c.critic_scheduler_step_size,c.scheduler_gamma)==(150,100,.5)
    assert (c.gradient_penalty_weight,c.critic_drift,c.velocity_clip,c.critic_clip)==(10.,.001,1000.,1e6)
    with pytest.raises(ValueError,match='unknown'):ExperimentConfig.from_dict({'surprise':1})


@pytest.mark.parametrize('kwargs',[{'method':'other'},{'device':'tpu'},{'epochs':0},{'batches':0},
    {'batch_size':0},{'critic_cycles':0},{'velocity_min':3000.,'velocity_max':2000.},
    {'velocity_lr':float('nan')},{'regularization':-1.},{'method':'fwi','noise':True},
    {'method':'fwi','train_source':True},{'method':'fwi','train_sigma':True},
    {'observation_mode':'normalized'},{'method':'fwigan','train_sigma':True,'noise':False}])
def test_invalid_config(kwargs):
    package()
    from fwigan.config import ExperimentConfig
    with pytest.raises(ValueError):ExperimentConfig(**kwargs)


def test_observation_portable_roundtrip_fingerprint_and_safe_overwrite(tmp_path):
    data=observations()
    from fwigan.data.io import save_observations,load_observations
    path=tmp_path/'data.pt';save_observations(path,data)
    loaded=load_observations(path)
    assert loaded.reference_velocity is None
    assert loaded.fingerprint()==data.fingerprint()
    assert loaded.to('cpu').fingerprint()==data.fingerprint()
    loaded.traces[0,0,0]=1.;assert loaded.fingerprint()!=data.fingerprint()
    with pytest.raises(FileExistsError):save_observations(path,data)
    artifact=torch.load(path,map_location='cpu');artifact['schema']=999;torch.save(artifact,tmp_path/'bad.pt')
    with pytest.raises(ValueError,match='schema'):load_observations(tmp_path/'bad.pt')


def test_tiny_data_real_solver_raw_units_and_reproducible_noise():
    package()
    from fwigan.data.synthetic import make_tiny_data
    from fwigan.physics import LegacyDeepwave
    clean=make_tiny_data(nt=64)
    assert clean.grid.shape==(40,70) and clean.grid.spacing==30. and clean.sampling.dt==.003
    assert clean.traces.shape==(2,60,64)
    assert torch.all(clean.initial_velocity==2000.) and torch.all(clean.reference_velocity==2100.)
    expected=LegacyDeepwave().predict(clean.reference_velocity,clean.source,clean.acquisition,clean.grid,clean.sampling)
    torch.testing.assert_close(clean.traces,expected)
    noisy=make_tiny_data(seed=42,noise=.05,nt=64)
    torch.testing.assert_close(noisy.traces,make_tiny_data(seed=42,noise=.05,nt=64).traces,rtol=0,atol=0)
    torch.manual_seed(42)
    torch.testing.assert_close(noisy.traces-clean.traces,.05*torch.randn_like(clean.traces),rtol=1e-5,atol=2e-5)


def test_distinct_initial_source_roundtrip_and_fingerprint(tmp_path):
    data=observations()
    f=package()
    from fwigan.data.io import save_observations,load_observations
    args=dict(traces=data.traces,acquisition=data.acquisition,grid=data.grid,sampling=data.sampling,
              source=data.source,initial_velocity=data.initial_velocity)
    initial_source=data.source*.5
    distinct=f.ObservationData(**args,initial_source=initial_source)
    assert distinct.fingerprint()!=data.fingerprint()
    path=tmp_path/'distinct.pt';save_observations(path,distinct)
    loaded=load_observations(path)
    torch.testing.assert_close(loaded.source,data.source)
    torch.testing.assert_close(loaded.initial_source,initial_source)
    assert loaded.fingerprint()==distinct.fingerprint()==distinct.to('cpu').fingerprint()
    with pytest.raises(FileExistsError):save_observations(path,data)
    for invalid in (initial_source.double(), initial_source[...,:2],initial_source.repeat(3,1,1)):
        with pytest.raises(ValueError):f.ObservationData(**args,initial_source=invalid)


def test_noisy_simulation_preserves_notebook_draw_mapping():
    f,grid,sampling,acq,source=specs()
    from fwigan.physics import LegacyDeepwave
    from fwigan.data.preprocessing import legacy_normalize,to_critic_layout
    velocity=torch.full(grid.shape,2000.)
    generator=Generator(velocity,torch.tensor(.3),source.permute(2,0,1),2,30.,.003,noise=True)
    torch.manual_seed(123)
    expected=generator(acq.sources,acq.receivers)
    torch.manual_seed(123)
    actual=to_critic_layout(legacy_normalize(LegacyDeepwave().predict(velocity,source,acq,grid,sampling)))
    actual=actual+.3*torch.randn_like(actual)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)


@pytest.mark.parametrize('spacing,origin,shape,cell,below', [
    (30., (0.,0.), (40,70), 10, True),
    (.7, (0.,0.), (40,70), 13, False),
    (.7, (1.4,2.8), (40,70), 13, False),
    (.7, (1.4,2.8), (40,14), 13, False),
])
def test_coordinates_truncated_to_previous_cell_are_rejected(spacing,origin,shape,cell,below):
    f=package()
    from fwigan.physics import LegacyDeepwave
    grid=f.GridSpec(shape,spacing,origin)
    point=torch.tensor(origin).reshape(1,1,2).clone()
    point[...,1]=torch.tensor(float(cell))*spacing+origin[1]
    if below:point[...,1]=torch.nextafter(point[...,1],torch.tensor(float('-inf')))
    relative=point-point.new_tensor(origin)
    native=dw.base.model.Model({'vp':torch.full(shape,20.)},spacing)
    assert native.get_locations(relative)[0,0,1].item()==cell-1
    acquisition=f.Acquisition(point,point.clone())
    source=dw.wavelets.ricker(100.,32,.0005,.01).reshape(1,1,-1)
    with pytest.raises(ValueError,match='unsupported.*coordinate|coordinate.*unsupported'):
        LegacyDeepwave().predict(torch.full(shape,20.),source,acquisition,grid,f.SamplingSpec(32,.0005))


def test_supported_fractional_spacing_origin_last_point_uses_intended_backend_cell():
    f=package()
    from fwigan.physics import LegacyDeepwave
    grid=f.GridSpec((40,70),.7,(1.4,2.8))
    point=torch.tensor(grid.origin).reshape(1,1,2).clone()
    point[...,1]=torch.tensor(69.)*grid.spacing+grid.origin[1]
    acquisition=f.Acquisition(point,point.clone())
    velocity=torch.full(grid.shape,20.)
    source=dw.wavelets.ricker(100.,32,.0005,.01).reshape(1,1,-1)
    sampling=f.SamplingSpec(32,.0005)
    actual=LegacyDeepwave().predict(velocity,source,acquisition,grid,sampling)
    relative=point-point.new_tensor(grid.origin)
    prop=dw.scalar.Propagator({'vp':velocity},grid.spacing,pml_width=grid.pml_width,survey_pad=None)
    assert prop.model.get_locations(relative).tolist()==[[[0,69]]]
    expected=prop(source.permute(2,0,1),relative,relative,sampling.dt).permute(1,2,0)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)


@pytest.mark.parametrize('name,parameter_enabled,config_enabled',[
    ('source',False,True),('source',True,False),('sigma',False,True),('sigma',True,False),
])
def test_parameter_groups_reject_both_directions_of_configuration_mismatch(name,parameter_enabled,config_enabled):
    f,grid,sampling,acq,source=specs()
    from fwigan.parameters import InversionParameters
    from fwigan.config import ExperimentConfig
    flags={'train_source':False,'train_sigma':False};flags[f'train_{name}']=parameter_enabled
    parameters=InversionParameters(torch.full(grid.shape,2000.),source,torch.tensor(1.),**flags)
    flags[f'train_{name}']=config_enabled
    config=ExperimentConfig(noise=True,**flags)
    with pytest.raises(ValueError,match=name):parameters.optimizer_groups(config)


@pytest.mark.parametrize('mutation',['double','rank','dimensions','empty','nonfinite','shots'])
def test_mutated_acquisition_revalidates_at_all_entry_points(mutation):
    data=observations()
    from fwigan.physics import LegacyDeepwave
    sources=data.acquisition.sources
    mutations={'double':sources.double(),'rank':sources[0],
               'dimensions':sources[...,:1],'empty':sources[:,:0],
               'nonfinite':torch.full_like(sources,float('nan')),'shots':sources[:1]}
    data.acquisition.sources=mutations[mutation]
    for operation in (data.validate,data.fingerprint,
        lambda:LegacyDeepwave().predict(data.initial_velocity,data.source,data.acquisition,data.grid,data.sampling)):
        with pytest.raises(ValueError):operation()
