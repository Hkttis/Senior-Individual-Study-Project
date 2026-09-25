"""Create a concise report and static comparison figures from completed runs."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from run_paper_script.ch5_sector_exact_control import OFFICIAL_METRICS, METRIC_LABELS


def report(out):
    phys=pd.read_csv(out/'physics_distdir/runs.csv')
    summary=pd.read_csv(out/'physics_distdir/summary.csv')
    paired=pd.read_csv(out/'physics_distdir/paired_comparison.csv')
    bfgs=pd.read_csv(out/'bfgs_full/runs.csv')
    diag=pd.read_csv(out/'bfgs_endpoint_diagnostics.csv')
    plt.rcParams.update({'font.size':10})
    fig,axes=plt.subplots(2,4,figsize=(16,8),constrained_layout=True)
    for ax,metric in zip(axes.flat,OFFICIAL_METRICS):
        stats=summary[summary.metric==metric].set_index('variant')
        mean=[stats.loc[v,'mean'] for v in ['sector','exact']]
        sd=[stats.loc[v,'sd'] for v in ['sector','exact']]
        ax.bar([0,1],mean,yerr=sd,capsize=4,color=['#0072B2','#E69F00'])
        ax.set_xticks([0,1],['Sector','Exact'])
        ax.set_title(METRIC_LABELS[metric],fontsize=10,wrap=True)
        ax.grid(axis='y',alpha=.2)
    fig.suptitle('PhysicsSim Dist + Dir | identical weights | no repulsion / model anchors\n100 matched seeds; bars = mean +/- sample SD',fontsize=16)
    for ext in ['png','svg']:fig.savefig(out/f'physics_distdir_metrics.{ext}',dpi=200)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),constrained_layout=True)
    modes=['sector','exact']
    success=[int(((bfgs.variant==v)&(bfgs.status=='ok')).sum()) for v in modes]
    axes[0].bar([0,1],success,color=['#0072B2','#E69F00'],label='Converged')
    axes[0].bar([0,1],100-np.array(success),bottom=success,color='#cccccc',label='Failed')
    for i,n in enumerate(success):axes[0].text(i,103,f'{n}/100 converged',ha='center')
    axes[0].set(xticks=[0,1],xticklabels=['Sector','Exact'],ylim=(0,115),ylabel='Runs',title='Original BFGS convergence rule')
    axes[0].legend(loc='center right')
    for v,col in zip(modes,['#0072B2','#E69F00']):
        frame=diag[diag.variant==v]
        axes[1].scatter(frame.seed,frame.gradient_norm_inf,s=18,c=col,label=v.capitalize())
    axes[1].axhline(1e-3,color='black',linestyle='--',label='gtol = 1e-3')
    axes[1].set(yscale='log',xlabel='Seed',ylabel='Final gradient infinity norm',title='Finite endpoint diagnostics (including failures)')
    axes[1].legend()
    fig.suptitle('Anchored BFGS Full | same Sector weights: alpha=1, beta=-0.5')
    for ext in ['png','svg']:fig.savefig(out/f'bfgs_convergence_diagnostic.{ext}',dpi=200)
    plt.close(fig)
    lines=['# Same-weight direction-control experiment results','',
        'Both experiments use alpha=1 and beta=-0.5 without HPO. The DistDir block sets repulsion to zero and creates no model anchors. Seeds 0–99 are matched.','',
        '## Anchored BFGS Full','',f'Sector converged in {success[0]}/100 runs; Exact converged in {success[1]}/100. There are no successful Exact/Sector pairs, so no converged paired-effect estimate is available. Failed endpoints are retained in runs.csv and bfgs_endpoint_diagnostics.csv, not treated as solutions.','',
        'The Exact failed endpoints all have 山 → 焉耆 as their shortest direction edge. This near-collision diagnostic accompanies large gradients and precision-loss termination; it is not a proof of the cause or a general impossibility result.','',
        '## PhysicsSim Dist + Dir','',
        '| Metric | Sector mean ± SD | Exact mean ± SD | Exact − Sector [95% paired CI] |',
        '|---|---:|---:|---:|']
    for metric in OFFICIAL_METRICS:
        s=summary[summary.metric==metric].set_index('variant');p=paired[paired.metric==metric].iloc[0]
        lines.append(f'| {METRIC_LABELS[metric]} | {s.loc["sector","mean"]:.6g} ± {s.loc["sector","sd"]:.6g} | {s.loc["exact","mean"]:.6g} ± {s.loc["exact","sd"]:.6g} | {p.exact_minus_sector:.6g} [{p.ci95_lo:.6g}, {p.ci95_hi:.6g}] |')
    lines += ['', 'CI: 10000 paired percentile-bootstrap resamples. PhysicsSim completion means the prescribed finite-step procedure completed; it does not establish KKT convergence.', '',
        '## Verification and files','',
        '- verification.json: endpoint metric recomputation, initialization pairing, anchor/translation checks, and reproduction of the legacy Sector-DistDir metrics.',
        '- protocol.json and executed_source_snapshot/: actual execution settings and code provenance.',
        '- bfgs_full/ and physics_distdir/: seed checkpoints, initial/final coordinates, all run records, summaries and paired statistics.',
        '- New modules are separate from the original Sector objective and existing experiment entry points.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(summary.to_string(index=False));print(paired.to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--outdir',required=True)
    report(Path(p.parse_args().outdir))
