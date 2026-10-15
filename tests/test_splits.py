import numpy as np
import pytest
from conftest import needs_cache

from gaudit.data.splits import make_splits, split_masks

pytestmark = needs_cache
S = [1, 2, 3, 4, 5]


@pytest.mark.parametrize("scheme", ["trialwise", "serieswise", "shuffled"])
def test_within_subject_partitions_disjoint_and_complete(pool, scheme):
    for sp in make_splits(pool.w, scheme, S, seed=0, n_repeats=2):
        tr, va, te = split_masks(pool.w, sp)
        mine = pool.w.subject == sp.test_subjects[0]
        assert not (tr & te).any() and not (tr & va).any() and not (va & te).any()
        assert ((tr | va | te) == mine).all(), "partition must cover all windows of the subject"


def test_trialwise_ratio(pool):
    sp = make_splits(pool.w, "trialwise", S, seed=0, n_repeats=1)[0]
    n = len(sp.train) + len(sp.val) + len(sp.test)
    assert abs(len(sp.train) / n - 0.7) < 0.02 and abs(len(sp.test) / n - 0.2) < 0.02


def test_serieswise_tests_every_series_once(pool):
    sps = [sp for sp in make_splits(pool.w, "serieswise", S, seed=0) if sp.test_subjects == (1,)]
    tested = np.concatenate([sp.test for sp in sps])
    assert len(tested) == len(np.unique(tested))
    assert set(tested) == set(100 + np.unique(pool.w.series[pool.w.subject == 1]))


def test_subjectwise_disjoint_and_complete(pool):
    sps = make_splits(pool.w, "subjectwise", S, seed=0)
    assert sorted(int(sp.test[0]) for sp in sps) == S
    for sp in sps:
        assert len(sp.val) == 2
        assert set(sp.train) | set(sp.val) | set(sp.test) == set(S)
        assert not set(sp.train) & set(sp.test) and not set(sp.val) & set(sp.test)
        tr, va, te = split_masks(pool.w, sp)
        assert (tr | va | te).all()


def test_splits_are_reproducible(pool):
    a = make_splits(pool.w, "trialwise", S, seed=3, n_repeats=2)
    b = make_splits(pool.w, "trialwise", S, seed=3, n_repeats=2)
    assert all((x.test == y.test).all() for x, y in zip(a, b))


def test_blocked_split_partitions_blocks(pool):
    from gaudit.data.splits import block_ids, blocked

    for b in (0.05, 1.0, 8.0):
        sp = blocked(pool.w, 2, seed=0, block_s=b)
        tr, va, te = split_masks(pool.w, sp)
        ids = block_ids(pool.w, b)
        assert not np.intersect1d(ids[tr], ids[te]).size
        assert ((tr | va | te) == (pool.w.subject == 2)).all()


def test_blocked_split_shares_fewer_trials_as_blocks_grow(pool):
    from gaudit.data.splits import blocked

    shared = []
    for b in (0.05, 1.0, 20.0):
        tr, _, te = split_masks(pool.w, blocked(pool.w, 2, seed=0, block_s=b))
        shared.append(np.intersect1d(pool.w.trial_id[tr], pool.w.trial_id[te]).size)
    assert shared[0] > shared[1] > shared[2] == 0  # 20 s blocks = whole trials
