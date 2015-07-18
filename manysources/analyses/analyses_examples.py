# coding=utf-8
"""Analyse experimental results."""
from collections import defaultdict
from itertools import product, izip
import h5py

from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from manysources.datasets import ManysourcesDataset, MANYSOURCES_DATA_ROOT, MANYSOURCES_MOLECULES
from manysources.experiments import ManysourcesResult
import os.path as op


# Everyting is about fighting superdumb rules deeply rooted in the chem comunity
# Validation valuation external only make sense when evaluation really comes from a different assay


################################
# Per-fold sizes, class proportions and AUCs
################################

def merge_cache_sizes_aucs(dset='bcrp',
                           feats='ecfps1',
                           model='logreg1',
                           expid=0,
                           lso=True):
    """
    Merges and caches the sizes and performance information for the folds in the specified experiment,
    returning a pandas dataframe with the informations in records format:
      (dset, feats, model, expid, lso, fold_num, test_size, pos_proportion, auc)
    """

    # Coordinates
    cache_file = op.join(MANYSOURCES_DATA_ROOT, 'results', 'merged_fold_sizes_aucs.h5')
    group_id = '/dset=%s/feats=%s/model=%s/expid=%d/lso=%r' % (dset, feats, model, expid, lso)

    # Cache funcion for readability
    def cache():
        with h5py.File(cache_file, 'a') as h5w:  # Using h5 as we do here is highly inefficient
                                                 # (just attributes, no locality)
                                                 # It allows for:
                                                 #   - incremental addition of folds
                                                 #   - easy merging with other HDF5 files we are creating
                                                 #     (just add these attributes)

            group = h5w.require_group(group_id)

            def add_fold(dset, feats, model, expid, fold_num, fold):
                try:
                    auc = fold.auc()
                    test_size = fold.test_size()
                    pos_proportion = fold.pos_proportion()
                    fold_group = group.require_group('fold=%d' % fold_num)
                    fold_group.attrs['test_size'] = test_size
                    fold_group.attrs['pos_proportion'] = pos_proportion
                    fold_group.attrs['auc'] = auc
                except:
                    print 'Failed', dset, feats, model, expid, lso, fold_num

            with ManysourcesResult(expid=expid, dset=dset, feats=feats, model=model) as res:
                print dset, expid, model, feats, lso
                cv = res.lsocv() if lso else res.crscv()
                if cv is None:
                    return None
                for fold_num, fold in enumerate(cv.folds()):
                    try:
                        add_fold(dset, feats, model, expid, fold_num, fold)
                    except:
                        print 'ERROR', dset, feats, model, expid

    # Recache - LAME
    recache = False
    if not op.isfile(cache_file):
        recache = True  # LAME
    else:
        with h5py.File(cache_file, 'r') as h5:
            if not group_id in h5:
                recache = True
    if recache:
        cache()

    # Read
    with h5py.File(cache_file, 'r') as h5:
        records = []
        for fold_id, group in h5[group_id].iteritems():
            records.append((dset, feats, model, expid, lso,
                            int(fold_id.split('=')[1]),
                            group.attrs['test_size'],
                            group.attrs['pos_proportion'],
                            group.attrs['auc']))
        return pd.DataFrame(records, columns=['dset', 'feats', 'model', 'expid', 'lso',
                                              'fold',
                                              'test_size', 'pos_proportion', 'auc'])


def cache_all_fold_sizes_aucs(drop_na=True):
    cache_file = op.join(MANYSOURCES_DATA_ROOT, 'results', 'merged_fold_sizes_aucs_bigdf.pickled')
    if not op.isfile(cache_file):
        dsets = MANYSOURCES_MOLECULES.keys()
        feats_models = (('logreg1', 'ecfps1'),
                        ('logreg3', 'ecfps1'),
                        ('rfc1', 'rdkdescs1'))
        expids = range(4096)
        dfs = []
        for dset, (model, feats), expid, lso in product(dsets, feats_models, expids, (True, False)):
            dfs.append(merge_cache_sizes_aucs(dset=dset, model=model, feats=feats, expid=expid, lso=lso))
        big_df = pd.concat(dfs, ignore_index=True).reset_index(drop=True)
        big_df.set_index(['dset', 'feats', 'model', 'expid', 'lso', 'fold'])
        big_df.to_pickle(cache_file,)
    df = pd.read_pickle(cache_file)
    if drop_na:
        return df.dropna(axis=0)
    return df


def average_perf_plot(df=None):
    if not df:
        df = cache_all_fold_sizes_aucs(drop_na=True)
    # Quick review of average perf
    plt.figure()
    df.groupby(['dset', 'model', 'feats', 'lso'])['auc'].mean().plot(kind='bar')
    plt.show()


def two_violins(df, dset='bcrp', model='logreg3', feats='ecfps1'):
    # LSO vs CRS AUC, violin plot (missing Flo's aesthetics)
    df = df[(df.dset == dset) & (df.model == model) & (df.feats == feats)]
    plt.figure()
    sns.violinplot(df.auc, df.lso)
    plt.show()

#df = cache_all_fold_sizes_aucs(drop_na=True)
#two_violins(df, dset='mutagenicity')
#exit(33)

def unbalancy_scatterplot(df, dset='bcrp', model='logreg3', feats='ecfps1'):
    # LSO vs CRS, proportion vs AUC, scatter plot (missing Flo's aesthetics)
    # I guess we can do this with a groupby...
    df = df[(df.dset == dset) & (df.model == model) & (df.feats == feats)]
    plt.figure()
    plt.scatter(x=df[df.lso]['pos_proportion'], y=df[df.lso]['auc'], color='r', label='LSO')
    plt.scatter(x=df[~df.lso]['pos_proportion'], y=df[~df.lso]['auc'], color='b', label='CRS')
    plt.axvline(x=df[~df.lso]['pos_proportion'].min(), color='k')
    plt.axvline(x=df[~df.lso]['pos_proportion'].max() - 0.035, color='k')
    plt.xlabel('Positive proportion in test set', fontsize=22)
    plt.ylabel('AUC', fontsize=22)
    plt.ylim((0, 1))
    plt.legend(fontsize=22)
    plt.tick_params(labelsize=16)
    plt.show()


def twoviolins_nooutliers(df, dset='bcrp', model='logreg3', feats='ecfps1', pos_proportions_min=0.4,
                          pos_proportions_max=0.8):
    # Filter out outliers or degenerated cases: "too imbalanced"
    df = df[(df.dset == dset) & (df.model == model) & (df.feats == feats)]
    balanced = df[(df['pos_proportion'] > pos_proportions_min) & (df['pos_proportion'] < pos_proportions_max)]
    plt.figure()
    sns.violinplot(balanced.auc, balanced.lso)
    plt.draw()


# N.B. use violinplot ax parameter and others to control ranges, fonts etc.
# Do a multiplot


# from rdkit.Chem import AllChem
# dset = ManysourcesDataset('bcrp')
# fold_de_mierda = ['Ahmed-Belkacem_2005', 'Ali-Versiani_2011', 'Feng_2008', 'Giannini_2008']
# molids = dset.mols().sources2molids(fold_de_mierda)
# mols = [dset.mols().molid2mol(molid) for molid in molids]
# ys = [dset.mols().molid2target(molid) for molid in molids]
# print molids
# print ys
# for molid, mol in zip(molids, mols):
#     print molid, AllChem.MolToSmiles(mol)
# exit(26)




def logistic_from_weights(weights, intercept):
    # Rebuild the trained model given the parameters
    logreg = LogisticRegression()
    logreg.coef_ = weights
    logreg.intercept_ = intercept
    return logreg


def density(vector, eps=1E-6):
    return np.sum(np.abs(vector) > eps) / float(len(vector))


#### HERE wE ARE NOT USING THE PARAMETER "SOURCE"...
def source_only_features(dset='bcrp',
                         model='logreg3',
                         feats='ecfps1',
                         expids=range(20),
                         source='phenylquinazolines_Juvale_2012'):
    """"""
    dset = ManysourcesDataset(dset)
    sparsities = defaultdict(list)
    for expid in expids:
        res = ManysourcesResult(expid=expid, dset=dset.name, feats=feats, model=model)
        # models
        lso_models = [logistic_from_weights(weights, intercept) for weights, intercept, _ in res.lsocv().all_models()]
        crs_models = [logistic_from_weights(weights, intercept) for weights, intercept, _ in res.crscv().all_models()]
        # is sparsity the same?
        for lsom, crsm in zip(lso_models, crs_models):
            sparsities['sparsity_lso'].append(density(lsom.coef_[0, :]))
            sparsities['sparsity_crs'].append(density(crsm.coef_[0, :]))
    return pd.DataFrame(sparsities)

#
# Not very competitive, but logreg3 gets high weight sparsity at reasonable performance
#
# molids, lso_losses = collect_losses(dset='hERG')
# _, crs_losses = collect_losses(dset='hERG', lso=False)
# data = {'source': ManysourcesDataset(dset='hERG').mols().sources(molids),
#         'molid': molids,
#         'lso_sqloss': lso_losses,
#         'crs_sqloss': crs_losses}
#
# df = pd.DataFrame(data)
# df['loss_diff'] = df['lso_sqloss'] - df['crs_sqloss']
#
# print df['loss_diff'].mean()
# per_source = df.groupby('source')
# mean_losses = per_source.mean()
# print mean_losses.sort('loss_diff')
# for source, group in per_source:
#     print '%s: %.4f (%d)' % (source, group['loss_diff'].mean(), len(group))
#
# exit(33)
#
# Tests for sources in lso
# assert set(chain(*coocs)) == dset.mols().present_sources(), 'All sources should be there'
# for s1, s2 in combinations(coocs, 2):
#     assert len(s1 & s2) == 0, 'No source should be repeated for LSO'
#

if __name__ == '__main__':
    dset = 'bcrp'
    feats = 'ecfps1'
    model = 'logreg3'
    lso = True

    # Pandas side of things
    print scores_df().columns

#
# TODO: rerun experiments splitting sources into sub-sources (will make more sound the coocurrences analysis)
# FIXME: Problem with scores at hERG rfc1 rdkdescs1, need to drop nans?
# FIXME: Also with scores at mutagenicity rdkit descs, need to drop nans?
#
