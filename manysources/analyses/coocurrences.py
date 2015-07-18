from itertools import product
import h5py
import numpy as np

from manysources.datasets import MANYSOURCES_DATA_ROOT, MANYSOURCES_MOLECULES
from manysources.experiments import ManysourcesResult
import os.path as op

#
# TODO: written in a hurry in California,
#       improve, test, document and make more efficient (e.g. symlink stuff to same dset and feats)
#


def molecules_coocurrences_df(dset='bcrp',
                              expids=None,
                              feats='ecfps1',
                              model='logreg3',
                              lso=True):
    cache_file = op.join(MANYSOURCES_DATA_ROOT, 'results', 'molecules_coocurrences_df.h5')
    if expids is None:
        expids = range(4096)
    group_id = '/dset=%s/feats=%s/model=%s/lso=%r' % (dset, feats, model, lso)
    molecules_dataset_id = '/dset=%s/molecules' % dset
    with h5py.File(cache_file, 'a') as h5:
        if group_id not in h5:
            coocurrences = []
            valid_expids = []
            fold_ids = []
            dset_name = dset
            dset = None
            for expid in expids:
                print dset_name, expid, model, feats, lso
                res = ManysourcesResult(expid=expid, dset=dset_name, feats=feats, model=model)
                if dset is None:
                    dset = res.ms_dset()
                if molecules_dataset_id not in h5:
                    molecules_as_in_the_matrix = res.molids()
                    h5[molecules_dataset_id] = molecules_as_in_the_matrix
                cv = res.lsocv() if lso else res.crscv()
                if cv is None:
                    continue
                for fold_num, fold in enumerate(cv.folds()):
                    try:
                        c = np.zeros(len(res.molids()), dtype=np.int)
                        c[fold.test_indices()] = 1
                        coocurrences.append(c)
                        valid_expids.append(expid)
                        fold_ids.append(fold_num)
                    except:
                        pass
                res._close_h5()
            group = h5.require_group(group_id)
            group['coocurrences'] = np.array(coocurrences)
            group['expids'] = valid_expids
            group['folds'] = fold_ids
    with h5py.File(cache_file, 'r') as h5:
        molids = h5[molecules_dataset_id][:]
        coocurrences = h5[group_id]['coocurrences'][:].astype(np.bool)
        expids = h5[group_id]['expids'][:]
        folds = h5[group_id]['folds'][:]
        return coocurrences, molids, expids, folds


def cache_all_molecules_coocurrences():
    dsets = MANYSOURCES_MOLECULES.keys()
    feats_models = (('logreg1', 'ecfps1'),
                    ('logreg3', 'ecfps1'),  # These two are the same, just put a link in the HDF5...
                    ('rfc1', 'rdkdescs1'))
    for dset, (model, feats) in product(dsets, feats_models):
        molecules_coocurrences_df(dset=dset, model=model, feats=feats, lso=True)
        molecules_coocurrences_df(dset=dset, model=model, feats=feats, lso=False)


def sources_coocurrences_df(dset='bcrp',
                            expids=None,
                            feats='ecfps1',
                            model='logreg3',
                            lso=True):
    cache_file = op.join(MANYSOURCES_DATA_ROOT, 'results', 'sources_coocurrences_df.h5')
    if expids is None:
        expids = range(4096)
    group_id = '/dset=%s/feats=%s/model=%s/lso=%r' % (dset, feats, model, lso)
    sources_dataset_id = '/dset=%s/sources' % dset
    with h5py.File(cache_file, 'a') as h5:
        if group_id not in h5:
            coocurrences = []
            valid_expids = []
            fold_ids = []
            dset_name = dset
            dset = None
            for expid in expids:
                print dset_name, expid, model, feats, lso
                res = ManysourcesResult(expid=expid, dset=dset_name, feats=feats, model=model)
                if dset is None:
                    dset = res.ms_dset()
                if sources_dataset_id not in h5:
                    sources_as_in_the_matrix = dset.mols().i2sources_order()
                    h5[sources_dataset_id] = sources_as_in_the_matrix
                if lso is False:
                    print '\tWARNING: source coocurrences do not make much sense without LSO splitting'
                cv = res.lsocv() if lso else res.crscv()
                if cv is None:
                    continue
                for fold_num, fold in enumerate(cv.folds()):
                    try:
                        coocurrences.append(fold.sources_coocurrences(dset=dset))
                        valid_expids.append(expid)
                        fold_ids.append(fold_num)
                    except:
                        pass
                res._close_h5()
            group = h5.require_group(group_id)
            group['coocurrences'] = np.array(coocurrences)
            group['expids'] = valid_expids
            group['folds'] = fold_ids
    with h5py.File(cache_file, 'r') as h5:
        sources = h5[sources_dataset_id][:]
        coocurrences = h5[group_id]['coocurrences'][:].astype(np.bool)
        expids = h5[group_id]['expids'][:]
        folds = h5[group_id]['folds'][:]
        return coocurrences, sources, expids, folds


def cache_all_sources_coocurrences():
    dsets = MANYSOURCES_MOLECULES.keys()
    feats_models = (('logreg1', 'ecfps1'),
                    ('logreg3', 'ecfps1'),  # These two are the same, just put a link in the HDF5...
                    ('rfc1', 'rdkdescs1'))
    for dset, (model, feats) in product(dsets, feats_models):
        sources_coocurrences_df(dset=dset, model=model, feats=feats, lso=True)
        sources_coocurrences_df(dset=dset, model=model, feats=feats, lso=False)

if __name__ == '__main__':
    dset = 'bcrp'
    feats = 'ecfps1'
    model = 'logreg3'
    lso = True
    coocurrences, molids, expids, folds = molecules_coocurrences_df(dset=dset, feats=feats, model=model, lso=lso)