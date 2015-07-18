# Extracted from "analyses_examples"

import os.path as op
import numpy as np
import pandas as pd
import h5py
from itertools import product, izip
from collections import defaultdict

from manysources.datasets import MANYSOURCES_DATA_ROOT, MANYSOURCES_MOLECULES
from manysources.experiments import ManysourcesResult


# --- CV scores merging

def merge_cache_scores(dset_id='bcrp',
                       expids=None,
                       feats='ecfps1',
                       model='logreg1',
                       lso=True,
                       calib=None):
    """
    Returns a 5-tuple (scores, expids, folds, molids, y) where:

      - scores is a num_mols x num_valid_expids matrix with the scores of each molecule in each (valid) experiment
      - expids is a num_valid_expids array of valid expids
      - folds is a num_mols x num_valid_expids matrix of fold assignment for each molecule in each (valid) experiment
      - molids is a num_mols array with the pertinent mol_ids
      - y is a num_mols array with the labels of the molecules
    """
    cache_file = op.join(MANYSOURCES_DATA_ROOT, 'results', 'scores_df.h5')
    if expids is None:
        expids = range(4096)
    group_id = '/dset=%s/feats=%s/model=%s/lso=%r/calib=%r' % (dset_id, feats, model, lso, calib)
    dset_feats_id = '/dset=%s/feats=%s' % (dset_id, feats)
    with h5py.File(cache_file, 'a') as h5:
        if group_id not in h5:
            group = h5.require_group(group_id)
            dset_feats_group = h5[dset_feats_id]
            scoress = []
            foldss = []
            correct_expids = []
            for expid in expids:
                print dset_id, expid, model, feats
                res = ManysourcesResult(expid=expid, dset=dset_id, feats=feats, model=model)
                cv = res.lsocv() if lso else res.crscv()
                try:
                    scores, ys, folds = cv.merge_scores(calibration=calib)
                    if 'y' not in dset_feats_group:
                        dset_feats_group['y'] = np.array(ys, dtype=np.int32)
                    if 'molids' not in dset_feats_group:
                        dset_feats_group['molids'] = res.molids()
                    scoress.append(scores)
                    foldss.append(folds)
                    correct_expids.append(expid)
                except:
                    pass
                finally:
                    res._close_h5()  # TODO: make result a context manager...
            group['scores'] = np.array(scoress).T
            group['folds'] = np.array(foldss).T
            group['expids'] = np.array(correct_expids, dtype=np.int32)

        with h5py.File(cache_file, 'r') as h5:
            group1 = h5[group_id]
            group2 = h5[dset_feats_id]
            try:
                return \
                    group1['scores'][:], \
                    group1['expids'][:], \
                    group1['folds'][:], \
                    group2['molids'][:], \
                    group2['y'][:]
            except:
                return None


def cache_all_scores(calib=None, lso=True):
    dsets = MANYSOURCES_MOLECULES.keys()
    feats_models = (('logreg1', 'ecfps1'),
                    ('logreg3', 'ecfps1'),
                    ('rfc1', 'rdkdescs1'))
    for dset, (model, feats) in product(dsets, feats_models):
        merge_cache_scores(dset_id=dset, model=model, feats=feats, calib=calib, lso=lso)


if __name__ == '__main__':
    cache_all_scores(lso=True, calib=None)