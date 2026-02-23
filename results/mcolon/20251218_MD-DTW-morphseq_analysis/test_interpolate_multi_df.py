import numpy as np
import pandas as pd
from pathlib import Path

from src.analyze.trajectory_analysis.trajectory_utils import interpolate_to_common_grid_multi_df


def make_sample_df():
    time_a = np.array([18.0, 19.0, 20.0, 21.0])
    time_b = np.array([19.5, 20.5, 21.5])

    rows = []
    for t in time_a:
        rows.append({'embryo_id': 'E1', 'predicted_stage_hpf': t, 'm1': t*1.0, 'm2': t*2.0})
    for t in time_b:
        rows.append({'embryo_id': 'E2', 'predicted_stage_hpf': t, 'm1': t*1.5, 'm2': t*2.5})

    return pd.DataFrame(rows)


def test_basic_interpolation():
    df = make_sample_df()
    out = interpolate_to_common_grid_multi_df(df, ['m1', 'm2'], grid_step=0.5, verbose=False)
    # Should have time column and m1/m2
    assert 'predicted_stage_hpf' in out.columns
    assert 'hpf' in out.columns
    assert 'm1' in out.columns and 'm2' in out.columns

    # Time grid should be from 18.0 to 21.5 step 0.5
    times = np.sort(out['predicted_stage_hpf'].unique())
    assert times[0] == 18.0 and times[-1] == 21.5


def test_fill_edges_flag():
    df = make_sample_df()
    # when fill_edges=False, E2 should have NaN at earliest times before 19.5
    out = interpolate_to_common_grid_multi_df(df, ['m1'], grid_step=0.5, fill_edges=False, verbose=False)
    e2 = out[out['embryo_id'] == 'E2'].sort_values('predicted_stage_hpf')
    # earliest timepoint 18.0 should not be present for E2 (masked)
    assert not (e2['predicted_stage_hpf'] == 18.0).any()

    # when fill_edges=True, E2 will have values at those edges (filled)
    out2 = interpolate_to_common_grid_multi_df(df, ['m1'], grid_step=0.5, fill_edges=True, verbose=False)
    e2b = out2[out2['embryo_id'] == 'E2']
    assert (e2b['predicted_stage_hpf'] == 18.0).any()


if __name__ == '__main__':
    test_basic_interpolation()
    test_fill_edges_flag()
    print('All interpolate_to_common_grid_multi_df tests passed')
