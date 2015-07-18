# coding=utf-8
from itertools import product
from time import time
import os.path as op
import matplotlib.pyplot as plt
import numpy as np

import h5py
import pandas as pd
from sklearn.metrics import roc_auc_score
from whatami import whatable

from manysources.datasets import MANYSOURCES_DATA_ROOT, ManysourcesDataset
from manysources.experiments import ManysourcesResult, DEFAULT_EXPIDS


@whatable
class CVScore(object):

    def __init__(self,
                 name,
                 is_loss=True,
                 per_mol=True,
                 columns=None):
        super(CVScore, self).__init__()
        self.name = name
        self._is_loss = is_loss
        self._per_mol = per_mol
        self._columns = columns

    def per_mol(self):
        return self._per_mol

    def is_loss(self):
        return self._is_loss

    def columns(self):
        if self._columns is None:
            return self.what().id(),
        return self._columns

    def _compute_one(self, scores, labels, folds):
        raise NotImplementedError()

    def compute(self, scores, labels, folds):
        if not self.per_mol():
            df = pd.DataFrame({'scores': scores,
                               'labels': labels,
                               'folds': folds})

            losses = [self._compute_one(fold_df.scores, fold_df.labels, fold_df.folds)
                      for fold, fold_df in df.sort('folds').groupby(folds)]
        else:
            return self._compute_one(scores, labels, folds)


class SquaredError(CVScore):
    def __init__(self):
        super(SquaredError, self).__init__(name='sqerr',
                                           is_loss=True,
                                           per_mol=True,
                                           columns=('sqerror',))

    def _compute_one(self, scores, labels, folds):
        return (labels - scores) ** 2


class ROCAUC(CVScore):

    def __init__(self):
        super(ROCAUC, self).__init__(name='rocauc',
                                     is_loss=False,
                                     per_mol=False,
                                     columns=('rocauc_mean', 'rocauc_std', 'rocauc_stacked'))

    def _compute_one(self, scores, labels, folds):
        return roc_auc_score(labels, scores)  # also tweak "average" and "sample_weight"


def read_losses(dset='bcrp',
                feats='ecfps1',
                model='logreg3',
                #expids=DEFAULT_EXPIDS,
                expids=tuple(range(4096)),
                calibration=None,
                lso=True,
                verbose=False,
                also_folds=False):
    """
    N.B. at the moment, only squared loss.
    """
    # the path to the cache file
    cache_path = op.join(MANYSOURCES_DATA_ROOT,
                         'results',
                         'square_losses.h5')
    # the path to the results group
    result_coords = '/dset=%s/feats=%s/model=%s/lso=%r/score_calibration=%r' % \
                    (dset, feats, model, lso, calibration)

    #
    # each group will have a few datasets:
    #   - molids: a string list, created once, with the molecule ids
    #   - expids: a growable dataset with the expid
    #   - losses: a growable 2 dimensional dataset with a mols per columns and a row per expid (pointing to loss)
    #   - mfolds: a growable 2 dimensional dataset with a mols per columns and a row per expid (pointing to foldid)
    #
    # Storage is write-once, read-many, no-delete
    #

    # Try to read
    def read():
        with h5py.File(cache_path, 'r') as h5:
            group = h5[result_coords]
            # do we have all the requested expids?
            infile_expids = group['expids'][()] if expids is not None else expids
            if 0 == len(set(expids) - set(infile_expids[:, 0])):
                e2r = {e: i for e, i in infile_expids if i >= 0}
                ok_expids = [expid for expid in expids if expid in e2r]
                rows = [e2r[expid] for expid in ok_expids]
                losses = group['losses'][rows]
                folds = group['folds'][rows]
                molids = group['molids'][:]
                df_losses = pd.DataFrame(losses, columns=molids, index=ok_expids)
                df_folds = None if not also_folds else pd.DataFrame(folds, columns=molids, index=ok_expids)
                return df_losses, df_folds

    def write():
        with h5py.File(cache_path, 'a') as h5:
            group = h5.require_group(result_coords)
            infile_expids = set(group['expids'][:]) if 'expids' in group else {}
            expidss = []
            oks = 0
            losses = []
            foldss = []
            molids = None
            for expid in expids:
                if verbose:
                    print expid, lso
                if expid in infile_expids:
                    if verbose:
                        print '\tAlready done, skipping...'
                    continue
                try:
                    # look for the results corresponding to the desired expid, lso
                    res = ManysourcesResult(expid=expid, dset=dset, feats=feats, model=model).lsocv() if lso else \
                        ManysourcesResult(expid=expid, dset=dset, feats=feats, model=model).crscv()
                    # Merge the "CV" scores to have one score per compound in the dataset
                    scores, labels, folds = res.merge_scores(calibration=calibration)
                    if verbose:
                        print roc_auc_score(labels, scores, average='samples')
                    losses.append((labels - scores) ** 2)
                    foldss.append(folds)
                    if molids is None:
                        molids = res.molids()
                    expidss.append((expid, len(infile_expids) + oks))
                    oks += 1
                except:
                    # We guess that this happens when the external set only contains one class, but we need to check
                    print 'Warning, had troubles with', expid, lso
                    expidss.append((expid, -1))
            # write molids - N.B. assume same for all of them, which is reasonable
            if 'molids' not in group:
                group['molids'] = molids
            # write expids index
            expids_dset = group.require_dataset('expids',
                                                shape=(len(infile_expids) + len(expidss), 2),
                                                dtype=np.int32,
                                                maxshape=(None, 2))
            expids_dset.resize((len(infile_expids) + len(expidss), 2))
            expids_dset[len(infile_expids):] = expidss
            # write losses
            losses_dset = group.require_dataset('losses',
                                                shape=(len(infile_expids) + len(losses), len(molids)),
                                                dtype=np.float64,
                                                maxshape=(None, len(molids)))
            losses_dset.resize((len(infile_expids) + len(losses), len(molids)))
            losses_dset[len(infile_expids):] = losses
            # write folds (should be optional)
            folds_dset = group.require_dataset('folds',
                                               shape=(len(infile_expids) + len(losses), len(molids)),
                                               dtype=np.int32,
                                               maxshape=(None, len(molids)))
            folds_dset.resize((len(infile_expids) + len(losses), len(molids)))
            folds_dset[len(infile_expids):] = foldss

    try:
        return read()
    except:
        write()
        return read()


def collect_all_lossess(lso=True):
    """
    Reads accross all our interesting results so far, and write down the losses DataFrame into a big h5 file
    """
    dsets = ('bcrp', 'hERG', 'mutagenicity', 'pgp-barbara', 'pgp-barbara-verapamil', 'pgp-cruciani')
    featss = ('ecfps1',)
    models = ('logreg1', 'logreg3')
    calibrations = (None, 'all', '0-1')  # 'max-cheating', 'other-folds',
    for dset, feats, model, calib in product(dsets, featss, models, calibrations):
        print dset, feats, model, calib
        start = time()
        read_losses(dset=dset, feats=feats, model=model, calibration=calib, lso=lso)
        print 'took %.2f seconds' % (time() - start)


def nicita_petits_plot():

    # Read-in the data (fast atm)
    losses, _ = read_losses()
    losses['source'] = ManysourcesDataset('bcrp').mols().molids2sources(losses.index)
    df_mean_loss = losses.groupby('source').mean()
    df_source_sizes = losses.groupby('source').size()

    # Readability
    sources = list(df_mean_loss.index)
    num_sources = len(sources)

    # Split between lso and crs columns
    df_lso = df_mean_loss[[col for col in df_mean_loss.columns if 'lso=True' in col]]
    df_crs = df_mean_loss[[col for col in df_mean_loss.columns if 'lso=False' in col]]

    # Sort by LSO mean loss
    order = np.argsort(df_lso.mean(axis=1))  # How to do indirect sorting in pandas?
    df_crs = df_crs.iloc[order]
    df_lso = df_lso.iloc[order]
    df_source_sizes = df_source_sizes.iloc[order]

    fig, (axl, axr) = plt.subplots(nrows=1, ncols=2, sharey=False)  # Could be true

    # Plot LSO and CRS in same subplot
    axl.errorbar(df_lso.mean(axis=1),
                 np.arange(num_sources),
                 xerr=df_lso.std(axis=1),
                 fmt='ok', ecolor='gray', alpha=0.5,
                 label='lso mean sqrl')
    axl.errorbar(df_crs.mean(axis=1),
                 np.arange(num_sources),
                 xerr=df_crs.std(axis=1),
                 fmt='ob', ecolor='blue', alpha=0.5,
                 label='crs mean sqrl')
    axl.set_xlabel('mean squared loss')
    axl.set_ylabel('source')
    axl.set_xlim([0, 1])
    axl.set_ylim([-1, num_sources + 1])
    axl.set_yticks(range(num_sources))
    axl.set_yticklabels(df_lso.index)
    axl.legend()

    # Plot differences of the mean
    axr.errorbar(df_crs.mean(axis=1) - df_lso.mean(axis=1),
                 np.arange(num_sources),
                 fmt='ok', ecolor='gray', alpha=0.5)
    axr.set_xlabel('difference crs - lso')
    axr.set_ylabel('molcount')
    axr.set_ylim([-1, num_sources + 1])
    axr.set_yticks(range(num_sources))
    axr.set_yticklabels(map(str, df_source_sizes))
    axr.vlines(0, 0, len(df_lso), linewidth=5, alpha=0.2)
    axr.set_xlim([-1, 1])


if __name__ == '__main__':
    collect_all_lossess(lso=False)
    # nicita_petits_plot()
