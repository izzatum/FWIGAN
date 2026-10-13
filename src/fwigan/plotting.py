"""Optional Matplotlib views using physical axes and raw-amplitude traces."""
from pathlib import Path
import numpy as np


def plot_results(data, result, output_dir=None):
    """Plot a public export_final result with shared velocity scale and units.

    Returns two figures so notebooks can display them. Save PNGs when an output
    directory is supplied. Matplotlib is loaded only when plotting is requested.
    """
    import matplotlib.pyplot as plt
    data.validate()
    velocity = result['velocity'].detach().cpu().numpy()
    prediction = result['predicted'].detach().cpu().numpy()
    if velocity.shape != data.grid.shape or prediction.shape != tuple(data.traces.shape):
        raise ValueError('plot result shapes must match the observation dataset')
    if not np.isfinite(velocity).all() or not np.isfinite(prediction).all():
        raise ValueError('plot results must contain finite values')
    models = [('Initial velocity',data.initial_velocity.detach().cpu().numpy()),('Inverted velocity',velocity)]
    if data.reference_velocity is not None:
        models.insert(0,('Reference velocity',data.reference_velocity.detach().cpu().numpy()))
    minimum = min(model.min() for _,model in models)
    maximum = max(model.max() for _,model in models)
    if minimum == maximum: minimum,maximum = minimum-1,maximum+1
    z,x = data.grid.origin
    nz,nx = data.grid.shape
    extent = (x,x+(nx-1)*data.grid.spacing,z+(nz-1)*data.grid.spacing,z)
    physical,axes = plt.subplots(1,len(models),figsize=(4.2*len(models),4),layout='constrained')
    for ax,(title,model) in zip(axes,models):
        im = ax.imshow(model,extent=extent,vmin=minimum,vmax=maximum,cmap='viridis',aspect='equal')
        ax.set(title=title,xlabel='Horizontal position x (m)',ylabel='Depth z (m)')
    physical.colorbar(im,ax=axes,label='Velocity (m/s)',shrink=.8)
    observed = data.traces[0].detach().cpu().numpy()
    residual = prediction[0]-observed
    times = np.arange(data.sampling.nt)*data.sampling.dt
    traces,(ax,record) = plt.subplots(1,2,figsize=(11,4),layout='constrained')
    receiver = min(29,observed.shape[0]-1)
    ax.plot(times,observed[receiver],label='Observed',color='black')
    ax.plot(times,prediction[0,receiver],label='Predicted',color='#0072b2')
    ax.set(title=f'Shot 1, receiver {receiver+1}',xlabel='Time (s)',ylabel='Pressure amplitude (a.u.)')
    ax.legend()
    maximum = max(float(np.abs(residual).max()),1e-12)
    im = record.imshow(residual.T,extent=(-.5,observed.shape[0]-.5,times[-1],0),
                       cmap='PuOr',vmin=-maximum,vmax=maximum,aspect='auto')
    record.set(title='Predicted − observed, shot 1',xlabel='Receiver index',ylabel='Time (s)')
    traces.colorbar(im,ax=record,label='Residual pressure (a.u.)')
    if output_dir is not None:
        destination = Path(output_dir); destination.mkdir(parents=True,exist_ok=True)
        physical.savefig(destination/'velocity.png',dpi=160,bbox_inches='tight')
        traces.savefig(destination/'traces.png',dpi=160,bbox_inches='tight')
    return physical,traces
