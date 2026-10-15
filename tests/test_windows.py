import numpy as np
from conftest import CACHE, needs_cache

from gaudit.data.cache import load_stream
from gaudit.data.targets import target_values
from gaudit.data.windows import make_windows, trial_mask

pytestmark = needs_cache


def test_windows_inside_trials_and_target_at_last_sample(data):
    sd = data.sd[1]
    w = make_windows(sd, trial_mask(sd, "kin"), win=50, stride=5)
    tix = np.searchsorted(sd.trial_id, w.trial_id)
    assert (sd.trial_id[tix] == w.trial_id).all()
    start, stop = sd.offset[tix], sd.offset[tix] + sd.length[tix]
    assert ((w.end - 49) >= start).all() and (w.end < stop).all(), "a window crosses a trial boundary"
    y = target_values(sd, w.end, ["force", "pos_z"])
    assert np.allclose(y[:, 0], sd.force[w.end]) and np.allclose(y[:, 1], sd.pos[w.end, 2])
    assert np.allclose(w.t_rel, (w.end - start) / 100)


def test_force_dataset_is_constant_surface_only(data):
    sd = data.sd[2]
    w = make_windows(sd, trial_mask(sd, "force"))
    tix = np.searchsorted(sd.trial_id, np.unique(w.trial_id))
    assert (sd.block_type[tix] == 1).all()


def test_trial_segments_are_disjoint_on_series_clock():
    st = load_stream(str(CACHE), 3)
    for s in np.unique(st["series"]):
        m = st["series"] == s
        lo, ln = st["seg_start_abs"][m], st["seg_len_s"][m]
        last_sample = lo[:-1] + ln[:-1] - 0.01  # time of each segment's last 100 Hz sample
        assert (lo[1:] > last_sample + 1e-6).all(), "trial segments overlap"
