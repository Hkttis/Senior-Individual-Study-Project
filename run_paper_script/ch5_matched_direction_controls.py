"""Same-weight direction controls: anchored BFGS and unanchored PhysicsSim.

Run from physics_simulation with -m. No HPO, Procrustes, test-driven selection,
or overwrite of older experiments. Per-seed JSON checkpoints preserve failures.
"""
from __future__ import annotations
import os
os.environ.setdefault('SDL_VIDEODRIVER','dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
import pandas as pd
from library.config import FILE_PATHS, refer_pos_sim
from library.data_io import load_ini_data_from_csv, uploading_ground_truth, uploading_directional_data, get_anchor_labels, get_test_site_labels
from library.initialization import generate_CHEN_initial_positions
from library.progressive_alignment import place_in_anchor_frame
from library.scipy_objective import build_current_objective, ObjectiveWeights
from library.scipy_exact_objective import build_current_exact_objective
from library.scipy_minimizer import run_bfgs
from library.physics import main_physics_simulation
from library.units import data_Li2sim
from run_paper_script.ch5_ablation_progressive import _evaluate, _target_positions_sim
from run_paper_script.ch5_sector_exact_control import OFFICIAL_METRICS, METRIC_LABELS


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_seed(block, seed, alpha, beta):
    _, vertices, dni, _, distances = load_ini_data_from_csv(FILE_PATHS)
    gt = uploading_ground_truth(vertices,dni)
    anchors = get_anchor_labels()
    tests = get_test_site_labels()
    targets = _target_positions_sim(dni,gt,'鄯善',refer_pos_sim)
    directions = uploading_directional_data()
    anchored = block == 'bfgs_full'
    np.random.seed(seed)
    v,d,_,initial,fixed = generate_CHEN_initial_positions(list(refer_pos_sim),
        anchors if anchored else [], [gt[dni[a]] for a in anchors] if anchored else [], anchor_label='鄯善')
    assert v == vertices and d == dni
    initial=np.asarray(initial,float)
    initial_sha=hashlib.sha256(initial.tobytes()).hexdigest()
    weights=ObjectiveWeights.from_physics_hpo(alpha=alpha,beta=beta)
    rows=[]
    for mode in ['sector','exact']:
        start=time.perf_counter()
        row={'variant':mode,'seed':seed,'block':block,'initial_sha256':initial_sha,
             'model_anchor_count':len(fixed),'status':'failed','error':''}
        points=None
        try:
            if anchored:
                factory=build_current_objective if mode=='sector' else build_current_exact_objective
                problem=factory(weights=weights)
                centered=initial-np.asarray(refer_pos_sim)
                np.testing.assert_allclose(centered[problem.anchor_indices],problem.anchor_coordinates,atol=1e-10,rtol=0)
                y0=problem.pack(centered)
                history=[]
                def record(y):
                    value,grad=problem.fun_and_jac(y)
                    history.append({'iteration':len(history),'objective':value,'gradient_norm_inf':float(np.linalg.norm(grad,np.inf))})
                record(y0)
                result=run_bfgs(y0,problem,callback=record)
                row.update({k:value for k,value in result.items() if k!='y_final'})
                row['objective_initial']=problem.fun(y0)
                row['history']=history
                if result['y_final'] is not None:
                    final=result['y_final'];record(final)
                    points=problem.unpack(final)+np.asarray(refer_pos_sim)
                    row['components_final']=asdict(problem.components(final))
                    row['max_anchor_error_sim']=float(np.max(np.abs(points[problem.anchor_indices]-initial[problem.anchor_indices])))
                ok=result['success']
            else:
                assert not fixed
                _,_,_,final=main_physics_simulation(vertices,dni,data_Li2sim(distances),initial.copy(),directions,[],
                    weights.distance,0.0,weights.direction,plot=False,directional_objective=mode)
                raw=np.asarray(final,float)
                row['raw_positions']=raw.tolist()
                points=place_in_anchor_frame(raw,dni,'鄯善',refer_pos_sim)
                delta=points-raw
                assert np.allclose(delta,delta[0],atol=1e-10,rtol=0)
                row['alignment_translation_sim']=delta[0].tolist()
                ok=True
            if points is not None:
                row.update(_evaluate(mode,seed,points,vertices,dni,data_Li2sim(distances),directions,tests,targets,distances))
                row['positions']=points.tolist()
            row['status']='ok' if ok else 'failed'
            row['error']='' if ok else str(row.get('failure_reason'))
        except Exception as exc:
            row['status']='failed';row['error']=f'{type(exc).__name__}: {exc}'
        row['elapsed_seconds']=time.perf_counter()-start
        rows.append(row)
    return {'block':block,'seed':seed,'vertices':vertices,'initial_positions':initial.tolist(),'runs':rows}


def aggregate(out, blocks):
    for block in blocks:
        folder=out/block
        rows=[];pos=[];hist=[];initial=[]
        for path in sorted(folder.glob('seed_*.json')):
            data=json.loads(path.read_text(encoding='utf8'))
            for label,p in zip(data['vertices'],data['initial_positions']):
                initial.append({'seed':data['seed'],'label':label,'x_y_up_sim':p[0],'y_y_up_sim':p[1]})
            for row in data['runs']:
                rows.append({k:v for k,v in row.items() if k not in ('positions','raw_positions','history','components_final')})
                for label,p in zip(data['vertices'],row.get('positions',[])):
                    pos.append({'variant':row['variant'],'seed':row['seed'],'label':label,'x_y_up_sim':p[0],'y_y_up_sim':p[1]})
                hist.extend({'variant':row['variant'],'seed':row['seed'],**h} for h in row.get('history',[]))
        runs=pd.DataFrame(rows)
        runs.to_csv(folder/'runs.csv',index=False,encoding='utf-8-sig')
        pd.DataFrame(pos).to_csv(folder/'positions.csv',index=False,encoding='utf-8-sig')
        pd.DataFrame(initial).to_csv(folder/'initial_positions.csv',index=False,encoding='utf-8-sig')
        if hist: pd.DataFrame(hist).to_csv(folder/'objective_history.csv',index=False)
        good=runs[runs.status=='ok']
        summaries=[];paired=[]
        for mode,group in good.groupby('variant'):
            for metric in OFFICIAL_METRICS:
                x=group[metric].to_numpy(float)
                summaries.append({'variant':mode,'metric':metric,'n':len(x),'mean':float(x.mean()),'sd':float(x.std(ddof=1)) if len(x)>1 else 0})
        for i,metric in enumerate(OFFICIAL_METRICS):
            if not len(good):continue
            pivot=good.pivot(index='seed',columns='variant',values=metric)
            if not {'exact','sector'}.issubset(pivot.columns):continue
            pivot=pivot.dropna(subset=['exact','sector'])
            if not len(pivot):continue
            dif=(pivot.exact-pivot.sector).to_numpy()
            rng=np.random.default_rng(20260905+i)
            boot=rng.choice(dif,(10000,len(dif)),replace=True).mean(axis=1)
            lo,hi=np.quantile(boot,[.025,.975])
            paired.append({'metric':metric,'n_pairs':len(dif),'exact_minus_sector':float(dif.mean()),'ci95_lo':lo,'ci95_hi':hi})
        pd.DataFrame(summaries).to_csv(folder/'summary.csv',index=False)
        pd.DataFrame(paired,columns=['metric','n_pairs','exact_minus_sector','ci95_lo','ci95_hi']).to_csv(folder/'paired_comparison.csv',index=False)
        counts=runs.groupby(['variant','status']).size().to_dict()
        (folder/'completion.json').write_text(json.dumps({str(k):int(v) for k,v in counts.items()},indent=2))
        print(block,counts,flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--outdir',required=True)
    parser.add_argument('--blocks',nargs='+',choices=['bfgs_full','physics_distdir'],default=['bfgs_full','physics_distdir'])
    parser.add_argument('--n-seeds',type=int,default=100)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--alpha',type=float,default=1.)
    parser.add_argument('--beta',type=float,default=-.5)
    args=parser.parse_args()
    if args.n_seeds < 1 or args.workers < 1:
        parser.error('--n-seeds and --workers must be positive')
    out=Path(args.outdir);out.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parents[1]
    sources=[Path(FILE_PATHS[k]) for k in ['chen_data','directional_data','ground_truth_path']]
    sources += [root/p for p in ['library/scipy_objective.py','library/scipy_exact_objective.py','library/scipy_minimizer.py','library/physics.py','library/initialization.py','library/progressive_alignment.py','library/geometry.py','library/metrics.py','library/config.py','run_paper_script/ch5_matched_direction_controls.py']]
    config={'blocks':args.blocks,'seeds':list(range(args.n_seeds)),'alpha':args.alpha,'beta':args.beta,
        'full_weights':asdict(ObjectiveWeights.from_physics_hpo(alpha=args.alpha,beta=args.beta)),
        'distdir_repulsion_weight':0,'distdir_model_anchors':0,'hpo':False,
        'coordinates':'x east, y north; 1 sim = 10 Li = 4.15 km; LCC from existing geometry protocol',
        'alignment':'BFGS: fixed numerical anchors centered on Shanshan, add [600,250] for evaluation; unanchored DistDir: posthoc Shanshan translation only, no rotation/reflection/scaling',
        'evaluation':'existing 8 held-out sites, original 44 sector direction observations; no test-site fitting',
        'bfgs':'full memory, analytic gradient, gtol=1e-3, maxiter=200*dimension; strict domain failures; no retry or tolerance relaxation',
        'physics':'1001 updates dt=.01, mass10, spring damping50, resistance10; sector/exact receive identical initial coordinates',
        'summary':'successful runs only; paired CI uses intersection of successful seeds; all failures and finite failed endpoints retained',
        'bootstrap':'10000 paired percentile replicates, seed 20260905+metric index',
        'source_sha256':{str(p.resolve()):sha(p) for p in sources}}
    cp=out/'protocol.json'
    if cp.exists() and json.loads(cp.read_text()) != config:
        raise ValueError('Refusing to reuse output with changed protocol or sources')
    cp.write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf8')
    for source in sources:
        target=out/'executed_source_snapshot'/source.relative_to(root)
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and sha(target)!=sha(source):
            raise ValueError(f'Existing source snapshot differs: {target}')
        if not target.exists():shutil.copy2(source,target)
    jobs=[]
    for block in args.blocks:
        (out/block).mkdir(exist_ok=True)
        for seed in range(args.n_seeds):
            if not (out/block/f'seed_{seed:03d}.json').exists():jobs.append((block,seed))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(run_seed,b,s,args.alpha,args.beta):(b,s) for b,s in jobs}
        for future in as_completed(futures):
            b,s=futures[future];data=future.result()
            (out/b/f'seed_{s:03d}.json').write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
            print(f'{b} seed {s}: '+', '.join(f"{r["variant"]}={r["status"]}" for r in data['runs']),flush=True)
    aggregate(out,args.blocks)


if __name__=='__main__':main()
