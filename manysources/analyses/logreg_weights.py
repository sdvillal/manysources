# coding=utf-8
"""Access to weights of each individual logistic regression."""
from itertools import product, izip
from array import array
import os.path as op

import h5py
from scipy.sparse import csr_matrix, coo_matrix
import numpy as np
import pandas as pd

from manysources.datasets import MANYSOURCES_DATA_ROOT, MANYSOURCES_MOLECULES
from manysources.experiments import ManysourcesResult
import numpy as np


def logreg_weights(dset='bcrp',
                   expids=None,
                   feats='ecfps1',
                   model='logreg3',
                   lso=True,
                   eps=1E-6):
    """
    Parameters
    ----------
    dset: string, default 'bcrp'
      The dataset id

    expids: int list, default None
      The experiment ids; if None, use from 0 to 4096

    feats: string, default 'ecfps1'
      The id of the feature set

    model: string, default 'logreg3'
      The id of the model used

    lso: boolean, default True
      Whether the experiment corresponds to a LSO or CRS partitioning scheme

    eps: float, default 0.000001
      Little number to be considered 0

    Returns
    -------
    A four-tuple (matrix, intercepts, expids, folds)
      matrix: csr sparse matrix (num_folds x num_features)
      intercepts: numpy array (num_folds)
      expids: num_folds experiment ids
      folds: num_folds array of fold_ids within each experiment

      Each row of the matrix corresponds to the tuple (expid, fold).

    :rtype: (scipy.sparse.csr_matrix, np.array, np.array, np.array)
    """
    cache_file = op.join(MANYSOURCES_DATA_ROOT, 'results', 'logreg_weights_df.h5')
    if expids is None:
        expids = range(4096)
    group_id = '/dset=%s/feats=%s/model=%s/lso=%r' % (dset, feats, model, lso)
    with h5py.File(cache_file, 'a') as h5:
        if group_id not in h5:
            is_sparse = eps is not None
            if not is_sparse:
                raise NotImplementedError()
            else:
                row = 0
                rows = array('I')
                cols = array('I')
                vals = array('d')
                intercepts = array('d')
                correct_expids = array('I')
                correct_folds = array('I')
                num_feats = None  # real nasty
                for expid in expids:
                    print dset, expid, model, feats, lso
                    res = ManysourcesResult(expid=expid, dset=dset, feats=feats, model=model)
                    cv = res.lsocv() if lso else res.crscv()
                    if cv is None:
                        continue
                    for fold_num, fold in enumerate(cv.folds()):
                        try:
                            coef, intercept = fold.model_data()
                            coef = coef[0]  # Lame
                            if num_feats is None:
                                num_feats = len(coef)
                            else:
                                assert num_feats == len(coef), 'astiness is all around me, and so the feelin is gross'
                            intercept = intercept[()]  # Lame
                            non_zero = np.where(np.abs(coef) > eps)[0]
                            density = float(len(non_zero)) / len(coef)
                            if density > 0.35:
                                print '\tWARNING: sparsity %.2f' % density
                            cols.extend(non_zero)
                            rows.extend([row] * len(non_zero))
                            vals.extend(coef[non_zero])
                            correct_expids.append(expid)
                            correct_folds.append(fold_num)
                            intercepts.append(intercept[0])
                            row += 1
                        except:
                            pass
                    res._close_h5()
                group = h5.require_group(group_id)
                matrix = coo_matrix((vals, (rows, cols))).tocsr()
                group['indices'] = matrix.indices
                group['indptr'] = matrix.indptr
                group['data'] = matrix.data
                group['shape'] = (matrix.shape[0], num_feats)  # nasty nasty
                group['expids'] = correct_expids
                group['folds'] = correct_folds
                group['intercepts'] = intercepts
    with h5py.File(cache_file, 'r') as h5:
        group = h5[group_id]
        matrix = csr_matrix((group['data'][:], group['indices'][:], group['indptr'][:]),
                            shape=group['shape'][:])
        return matrix, group['intercepts'][:], group['expids'][:], group['folds'][:]


def logreg_weights_df(dset='bcrp',
                      expids=None,
                      feats='ecfps1',
                      model='logreg3',
                      lso=True,
                      eps=1E-6):

    # DO NOT USE, INEFFICIENT

    matrix, intercepts, expids, folds = logreg_weights(dset=dset,
                                                       expids=expids,
                                                       feats=feats,
                                                       model=model,
                                                       lso=lso,
                                                       eps=eps)

    matrix = matrix.tocsc()

    # http://stackoverflow.com/questions/17818783/populate-a-pandas-sparsedataframe-from-a-scipy-sparse-matrix
    # But this is slow as hell... leave some prints in there to show it
    columns = range(matrix.shape[1])
    index = ['expid=%d#fold=%d' % (expid, fold) for expid, fold in izip(expids, folds)]
    df = pd.SparseDataFrame(columns=columns, index=index, default_fill_value=0)
    for column in columns:
        print column
        # columns.append(pd.SparseSeries(matrix[:, column].toarray().ravel()))
        df[column] = pd.SparseSeries(matrix[:, column].toarray().ravel(), fill_value=0)
    return df


def cache_all_logreg_weights():
    dsets = MANYSOURCES_MOLECULES.keys()
    feats_models = (('logreg1', 'ecfps1'),
                    ('logreg3', 'ecfps1'))
    for dset, (model, feats) in product(dsets, feats_models):
        logreg_weights(dset=dset, model=model, feats=feats, lso=False)
        logreg_weights(dset=dset, model=model, feats=feats, lso=False)


if __name__ == '__main__':
    cache_all_logreg_weights()
