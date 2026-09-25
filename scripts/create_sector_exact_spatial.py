"""Visualize stored sector/exact endpoints without rerunning models."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import colors
from scripts.create_manuscript_spatial_comparisons import (
    _save_group_figure, _sha256, SELECTION_METRICS,
)
from scripts.create_section_6_5_visual_prototype import (
    _distance_edge_errors, _wrong_direction_nodes, _combined_overlay_extent,
)
from library.config import FILE_PATHS, refer_pos_sim, km2pix
from library.data_io import load_ini_data_from_csv, uploading_ground_truth
from library.units import data_Li2sim
from run_paper_script.ch5_ablation_progressive import _target_positions_sim


def main():
    root = Path(__file__).resolve().parents[1]
    exact = root / 'outputs/ch5_sector_exact_control_hpo10_final100'
    sector = root / 'outputs/ch5_progressive_as_physics_alpha_1_beta_-0.5_dc_alpha_-2_wang_current_100seeds_random1000_20260721'
    out = exact / 'spatial_visualization'
    out.mkdir(exist_ok=True)
    runs_path = exact / 'sector_exact_runs_by_seed.csv'
    runs = pd.read_csv(runs_path)
    cfg = json.loads((exact / 'gridsearch_config.json').read_text())
    _, vertices, dni, _, distances = load_ini_data_from_csv(FILE_PATHS)
    targets = _target_positions_sim(dni, uploading_ground_truth(vertices, dni), '鄯善', refer_pos_sim)
    records, points, selection, exports, site_errors = [], {}, [], [], []
    paths = [runs_path, exact / 'selected_final_positions_y_up_sim.csv', sector / 'progressive_final_positions_y_up_sim.csv']
    for variant, label, path in [
        ('PhysicsSim-Full', 'Sector', paths[2]),
        ('PhysicsSim-ExactDir', 'Exact-direction', paths[1]),
    ]:
        group = runs.loc[(runs.variant == variant) & (runs.status == 'ok')].copy()
        assert len(group) == 100
        values = group[list(SELECTION_METRICS)]
        med = values.median()
        mad = (values - med).abs().median()
        # Near-constant direction metrics must not amplify floating-point noise.
        active = (values.max() - values.min()) > 1e-10
        scale = mad.where(mad > 1e-10, 1.0)
        score = np.sqrt((((values - med) / scale).loc[:, active] ** 2).sum(axis=1))
        row = group.loc[score.idxmin()]
        seed = int(row.seed)
        frame = pd.read_csv(path)
        frame = frame.loc[frame.seed == seed]
        if 'variant' in frame:
            frame = frame.loc[frame.variant == variant]
        assert len(frame) == len(vertices) and frame.label.nunique() == len(vertices)
        p = frame.set_index('label').loc[vertices, ['x_y_up_sim','y_y_up_sim']].to_numpy(float)
        assert np.isfinite(p).all()
        name = f'{label} (seed {seed})'
        metrics = {k: float(row[k]) for k in SELECTION_METRICS}
        errors = [float(np.linalg.norm(p[dni[t]] - targets[t])/km2pix) for t in cfg['test_labels']]
        recomputed = float(np.sqrt(np.mean(np.square(errors))))
        assert np.isclose(recomputed, metrics['RMSE_test_km'], atol=1e-7)
        points[name] = p
        records.append({'variant':name, 'seed':seed, 'rerun_metrics':metrics})
        selection.append({'variant':variant, 'seed':seed, 'metrics':metrics,
            'selection_rule':'minimum MAD-standardized distance to model-specific median of four primary metrics; exclude ranges <= 1e-10',
            'active_metrics':list(values.columns[active]), 'score':float(score.loc[row.name]),
            'recomputed_rmse_km':recomputed})
        exports.extend({'variant':variant,'seed':seed,'label':v,'x_y_up_sim':p[i,0],'y_y_up_sim':p[i,1]} for i,v in enumerate(vertices))
        site_errors.extend({'variant':variant,'seed':seed,'label':t,'error_km':e} for t,e in zip(cfg['test_labels'], errors))
    edges = {v:_distance_edge_errors(p, data_Li2sim(distances), dni) for v,p in points.items()}
    wrong = {v:_wrong_direction_nodes(p, vertices, dni) for v,p in points.items()}
    _save_group_figure(records=records, points_by_variant=points, targets=targets,
        vertices=vertices, dni=dni, anchors=cfg['anchor_labels'], tests=cfg['test_labels'],
        edge_errors=edges, wrong_nodes=wrong,
        overlay_extent=_combined_overlay_extent(points, targets, dni, cfg['anchor_labels'],cfg['test_labels'],pad_frac=.10),
        overlay_norm=colors.Normalize(0,max(r['error_km'] for r in site_errors)),
        edge_norm=colors.Normalize(0,max(e for rows in edges.values() for _,_,e in rows)),
        output_stem=out/'sector_exact_spatial_reconstruction')
    pd.DataFrame(exports).to_csv(out/'plotted_positions.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(site_errors).to_csv(out/'representative_site_errors.csv',index=False,encoding='utf-8-sig')
    metadata={'selection':selection,'source_sha256':{str(p):_sha256(p) for p in paths},
        'top_row':'shared extent and error colour scale; archaeological test-site overlay',
        'bottom_row':'independent equal-aspect extents for readability; shared full-range distance-error colour scale; orange crosses mark endpoints of violated direction observations',
        'models_rerun':False}
    (out/'visualization_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(selection,ensure_ascii=False))


if __name__ == '__main__':
    main()
