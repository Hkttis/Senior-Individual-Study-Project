"""Plot existing same-seed BFGS endpoints, explicitly labelling failed runs.

No optimization is run. Seed 0 is the fixed default, not selected by RMSE.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import colors
from library.config import FILE_PATHS, refer_pos_sim, km2pix
from library.data_io import load_ini_data_from_csv, uploading_ground_truth, get_anchor_labels, get_test_site_labels
from library.units import data_Li2sim
from run_paper_script.ch5_ablation_progressive import _target_positions_sim
from scripts.create_manuscript_spatial_comparisons import _save_group_figure, _sha256, SELECTION_METRICS
from scripts.create_section_6_5_visual_prototype import _distance_edge_errors, _wrong_direction_nodes, _combined_overlay_extent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--experiment',default='outputs/ch5_matched_direction_controls_100seeds_20260905')
    args=parser.parse_args()
    experiment=Path(args.experiment)
    source=experiment/'bfgs_full'/f'seed_{args.seed:03d}.json'
    data=json.loads(source.read_text(encoding='utf8'))
    _,vertices,dni,_,distances=load_ini_data_from_csv(FILE_PATHS)
    assert vertices==data['vertices']
    assert len({r['initial_sha256'] for r in data['runs']})==1
    anchors,tests=get_anchor_labels(),get_test_site_labels()
    targets=_target_positions_sim(dni,uploading_ground_truth(vertices,dni),'鄯善',refer_pos_sim)
    records=[];points={};errors=[];metadata=[]
    for mode in ['sector','exact']:
        row=next(r for r in data['runs'] if r['variant']==mode)
        x=np.asarray(row['positions'],float)
        assert x.shape==(len(vertices),2) and np.isfinite(x).all()
        state='converged' if row['status']=='ok' else 'NOT CONVERGED'
        label=f'{mode.capitalize()} BFGS: {state}'
        metrics={m:float(row[m]) for m in SELECTION_METRICS}
        site_errors=[float(np.linalg.norm(x[dni[t]]-targets[t])/km2pix) for t in tests]
        assert np.isclose(np.sqrt(np.mean(np.square(site_errors))),metrics['RMSE_test_km'],atol=1e-7)
        for a in anchors:np.testing.assert_allclose(x[dni[a]],targets[a],atol=1e-9,rtol=0)
        points[label]=x
        records.append({'variant':label,'seed':args.seed,'rerun_metrics':metrics})
        errors.extend({'variant':mode,'seed':args.seed,'label':t,'error_km':e} for t,e in zip(tests,site_errors))
        metadata.append({'variant':mode,'seed':args.seed,'status':row['status'],'gradient_norm_inf':row['gradient_norm'],
            'failure_reason':row.get('failure_reason'),'metrics':metrics})
    edge_errors={v:_distance_edge_errors(p,data_Li2sim(distances),dni) for v,p in points.items()}
    wrong={v:_wrong_direction_nodes(p,vertices,dni) for v,p in points.items()}
    out=experiment/'bfgs_spatial_visualization';out.mkdir(exist_ok=True)
    stem=out/f'bfgs_sector_exact_seed{args.seed}_endpoints'
    _save_group_figure(records=records,points_by_variant=points,targets=targets,vertices=vertices,dni=dni,
        anchors=anchors,tests=tests,edge_errors=edge_errors,wrong_nodes=wrong,
        overlay_extent=_combined_overlay_extent(points,targets,dni,anchors,tests,pad_frac=.1),
        overlay_norm=colors.Normalize(0,max(r['error_km'] for r in errors)),
        edge_norm=colors.Normalize(0,max(e for rows in edge_errors.values() for _,_,e in rows)),output_stem=stem)
    pd.DataFrame(errors).to_csv(out/f'seed{args.seed}_site_errors.csv',index=False,encoding='utf-8-sig')
    info={'selection':'fixed seed requested by CLI; default first seed 0, not chosen by outcome',
        'source':str(source.resolve()),'sha256':_sha256(source),'runs':metadata,'optimization_rerun':False,
        'scope':'Exact panel is a failed numerical endpoint, not a converged reconstruction',
        'coordinates':'stored y-up sim coordinates, no additional alignment; shared overlay extent and colour scales; independent equal-aspect lower panels'}
    (out/f'seed{args.seed}_metadata.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(info,ensure_ascii=False))


if __name__=='__main__':main()
