"""Re-evaluate saved endpoints and verify pairing, alignment and provenance."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from library.config import FILE_PATHS, refer_pos_sim
from library.data_io import load_ini_data_from_csv, uploading_ground_truth, uploading_directional_data, get_test_site_labels
from library.scipy_objective import build_current_objective
from library.scipy_exact_objective import build_current_exact_objective
from run_paper_script.ch5_ablation_progressive import _evaluate, _target_positions_sim
from run_paper_script.ch5_sector_exact_control import OFFICIAL_METRICS
from library.units import data_Li2sim


def verify(out):
    root=Path(__file__).resolve().parents[1]
    cfg=json.loads((out/'protocol.json').read_text(encoding='utf8'))
    assert cfg['alpha']==1 and cfg['beta']==-.5
    snapshot=out/'executed_source_snapshot'
    for raw,expected in cfg['source_sha256'].items():
        p=Path(raw)
        saved=snapshot/p.relative_to(root) if snapshot.exists() else p
        assert hashlib.sha256(saved.read_bytes()).hexdigest()==expected, str(saved)
    # Confirm the final additive module matches the in-memory executed kernel.
    if snapshot.exists():
        spec=importlib.util.spec_from_file_location('executed_sector_objective',snapshot/'library/scipy_objective.py')
        old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
        if hasattr(old.FixedAnchorObjective,'_direction_residual'):
            class ExecutedExact(old.FixedAnchorObjective):
                def _direction_residual(self,angles):return np.abs(angles)
        else:
            exact_spec=importlib.util.spec_from_file_location('executed_exact_objective',snapshot/'library/scipy_exact_objective.py')
            exact_module=importlib.util.module_from_spec(exact_spec)
            sys.modules[exact_spec.name]=exact_module;exact_spec.loader.exec_module(exact_module)
            ExecutedExact=exact_module.ExactDirectionObjective
        base=old.build_current_objective()
        executed=ExecutedExact(vertices=base.vertices,distance_pairs=base.distance_pairs,distance_targets=base.distance_targets,
            direction_pairs=base.direction_pairs,direction_vectors=base.direction_vectors,direction_half_widths=base.direction_half_widths,
            anchor_positions=dict(zip(base.anchor_indices,base.anchor_coordinates)),weights=base.weights,epsilon=base.epsilon)
        new=build_current_exact_objective()
        for seed in range(100):
            y=np.random.default_rng(seed).normal(0,80,new.dimension)
            assert executed.fun(y)==new.fun(y)
            np.testing.assert_array_equal(executed.jac(y),new.jac(y))
    _,vertices,dni,_,distances=load_ini_data_from_csv(FILE_PATHS)
    directions=uploading_directional_data();tests=get_test_site_labels()
    targets=_target_positions_sim(dni,uploading_ground_truth(vertices,dni),'鄯善',refer_pos_sim)
    old_runs=pd.read_csv(root/'outputs/ch5_progressive_as_physics_alpha_1_beta_-0.5_dc_alpha_-2_wang_current_100seeds_random1000_20260721/progressive_runs_by_seed.csv')
    old_runs=old_runs[old_runs.variant=='PhysicsSim-DistDir'].set_index('seed')
    checks={};diagnostics=[]
    for block in cfg['blocks']:
        files=sorted((out/block).glob('seed_*.json'))
        assert len(files)==len(cfg['seeds'])
        max_error=0.;max_legacy=0.;counts={}
        for path in files:
            data=json.loads(path.read_text(encoding='utf8'))
            assert data['vertices']==vertices
            initial=np.asarray(data['initial_positions'],float)
            initial_hash=hashlib.sha256(initial.tobytes()).hexdigest()
            assert {r['variant'] for r in data['runs']}=={'sector','exact'}
            for row in data['runs']:
                assert row['initial_sha256']==initial_hash
                key=row['variant']+'/'+row['status'];counts[key]=counts.get(key,0)+1
                if 'positions' not in row:continue
                x=np.asarray(row['positions'],float)
                metrics=_evaluate(row['variant'],row['seed'],x,vertices,dni,data_Li2sim(distances),directions,tests,targets,distances)
                for m in OFFICIAL_METRICS:
                    err=abs(metrics[m]-row[m]);max_error=max(max_error,err)
                    assert np.isclose(metrics[m],row[m],rtol=1e-10,atol=1e-9),(path,m)
                if block=='physics_distdir':
                    assert row['model_anchor_count']==0
                    delta=x-np.asarray(row['raw_positions'])
                    np.testing.assert_allclose(delta,np.tile(delta[0],(len(x),1)),rtol=0,atol=1e-10)
                    np.testing.assert_allclose(x[dni['鄯善']],refer_pos_sim,rtol=0,atol=1e-10)
                    if row['variant']=='sector':
                        for m in OFFICIAL_METRICS:
                            diff=abs(row[m]-old_runs.loc[row['seed'],m]);max_legacy=max(max_legacy,diff)
                            assert diff<1e-9,(row['seed'],m,diff)
                else:
                    problem=(build_current_objective if row['variant']=='sector' else build_current_exact_objective)()
                    centered=x-np.asarray(refer_pos_sim)
                    np.testing.assert_allclose(centered[problem.anchor_indices],problem.anchor_coordinates,rtol=0,atol=1e-10)
                    # Reconstruct saved free variables from the final X; subtraction may
                    # magnify roundoff near direction-edge collisions.
                    y=problem.pack(centered);value,grad=problem.fun_and_jac(y)
                    assert np.isclose(value,row['objective_final'],rtol=1e-8,atol=1e-5)
                    if row['status']=='ok':
                        assert row['success'] and row['gradient_norm']<=1e-3
                        assert np.linalg.norm(grad,np.inf)<=1.01e-3
                    u,v=problem.direction_pairs.T
                    lengths=np.linalg.norm(x[v]-x[u],axis=1);j=int(np.argmin(lengths))
                    diagnostics.append({'variant':row['variant'],'seed':row['seed'],'status':row['status'],
                        'gradient_norm_inf':row['gradient_norm'],'RMSE_test_km':row['RMSE_test_km'],
                        'closest_direction_pair':vertices[u[j]]+' -> '+vertices[v[j]],
                        'min_direction_distance_sim':float(lengths[j]),'failure_reason':row.get('failure_reason')})
        checks[block]={'n_seeds':len(files),'counts':counts,'max_metric_recompute_error':max_error,'max_legacy_sector_metric_error':max_legacy}
    pd.DataFrame(diagnostics).to_csv(out/'bfgs_endpoint_diagnostics.csv',index=False,encoding='utf-8-sig')
    result={'status':'passed','checks':checks,'executed_snapshot_hashes_match':True,
        'additive_exact_kernel_matches_executed_kernel':True,
        'note':'Verification passing establishes data/protocol consistency, not optimizer convergence.'}
    (out/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--outdir',required=True)
    verify(Path(p.parse_args().outdir))
