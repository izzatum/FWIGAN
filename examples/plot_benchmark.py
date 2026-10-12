"""Render recorded benchmark samples; never synthesize performance results.

python examples/plot_benchmark.py --input docs/assets/tiny-cpu-benchmark.json
"""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input',type=Path,required=True)
parser.add_argument('--output-dir',type=Path,default=Path('docs/assets'))
args=parser.parse_args()
report=json.loads(args.input.read_text())
if report.get('schema')!=1 or not report.get('workloads'):
    parser.error('expected successful benchmark schema 1 with recorded workloads')
if report.get('device')!='cpu' or report.get('suite')!='tiny':
    parser.error('this figure renders the tiny CPU suite; GPU plots need separate labels')
args.output_dir.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':11,'font.family':'DejaVu Sans','svg.fonttype':'none',
                     'axes.spines.top':False,'axes.spines.right':False})
fig,(timing,counts)=plt.subplots(1,2,figsize=(12,4.4),layout='constrained')
labels=list(report['workloads'])
positions=np.arange(len(labels))
for offset,kind,label,color in [(-.18,'original','Original frozen loop','#6b7280'),(.18,'optimized','Package strategy','#0072b2')]:
    medians=[report['workloads'][name]['median_seconds_per_batch'][kind] for name in labels]
    timing.bar(positions+offset,medians,width=.32,label=label,color=color,alpha=.8)
    for position,name in zip(positions+offset,labels):
        samples=[sample['seconds_per_batch'] for sample in report['workloads'][name]['windows'][kind]]
        timing.scatter(position+np.linspace(-.08,.08,len(samples)),samples,s=24,color=color,edgecolors='white',linewidth=.6,zorder=3)
    adjoints=[report['workloads'][name]['untimed_counts'][kind]['solver_adjoint']/report['workloads'][name]['config']['batches'] for name in labels]
    counts.bar(positions+offset,adjoints,width=.32,color=color,alpha=.8)
    for position,value in zip(positions+offset,adjoints):counts.text(position,value+.1,f'{value:g}',ha='center',fontsize=11)
for ax in (timing,counts):
    ax.set_xticks(positions,[label.replace('tiny-','').capitalize() for label in labels])
    ax.set_axisbelow(True);ax.grid(axis='y',alpha=.2)
timing.set(title='Measured samples and median wall time',ylabel='Seconds per shot group')
timing.legend(frameon=False,fontsize=9)
counts.set(title='Actual DeepWave adjoints per shot group',ylabel='Adjoint calls',ylim=(0,max(counts.get_ylim()[1],8)))
cycles='/'.join(str(value) for value in sorted({work['config']['critic_cycles'] for work in report['workloads'].values()}))
fig.suptitle(f'Tiny synthetic {report["device"].upper()} benchmark · {cycles} critic cycles\nEquivalent warmed optimizer work; no full-scale speedup claim',fontsize=13)
for suffix in ('png','svg'):
    destination=args.output_dir/f'tiny-cpu-benchmark.{suffix}'
    fig.savefig(destination,dpi=190,bbox_inches='tight')
    if suffix=='svg':destination.write_text('\n'.join(line.rstrip() for line in destination.read_text().splitlines())+'\n')
plt.close(fig)
print(args.output_dir/'tiny-cpu-benchmark.png')
