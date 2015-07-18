from functools import partial
import os.path as op
from collections import defaultdict
from joblib import Parallel, cpu_count, delayed

import numpy as np
from rdkit.Chem.rdmolfiles import MolToSmiles
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
import seaborn
from rdkit.Chem import AllChem

from matplotlib import pyplot as plt
from matplotlib import gridspec
import pandas as pd
from chemdeco.pillow_utils import rdkit2im, artdeco2, artdeco1

from manysources.common.misc import ensure_dir
from manysources.datasets import MANYSOURCES_DATA_ROOT
from manysources.analyses.losses import read_losses
from manysources.analyses.cooccurrences import molecules_coocurrences_df, sources_coocurrences_df
from manysources.datasets import ManysourcesDataset
from manysources.analyses.better_or_worse_coocc import cv_splits


def get_y(molid, dset, feats, model, calibration=None, lso=True):
    """
    Given a molid and experiment coordinates to retrieve the loss matrix, returns a numpy array of all the losses
    for the given molid (4095 expids)
    """
    df_losses, _ = read_losses(dset=dset, feats=feats, model=model, calibration=calibration, lso=lso)
    y = np.array(df_losses[molid])
    expids = df_losses.index
    return y, expids


def get_X(molid, expids, dset, feats, model, lso=True):
    """
    Given a molid and an experiment coordinate, retrieves the matrix of cooccurrences for the folds when the molid
    was in test
    """
    cooc, molids, _, _ = molecules_coocurrences_df(dset=dset, feats=feats, model=model, lso=lso)
    index_of_molid = np.where(molids == molid)[0][0]
    col = cooc[:, index_of_molid]  # the column on which we put the condition
    interesting_Cooc = cooc[col]  # the matrix X
    # filter out the rows where we had troubles validating the model
    X = interesting_Cooc[expids, :]
    X = np.array(X, dtype=np.int)
    return X


def get_X_source(molid, expids, dset, feats, model, lso=True):
    """
    Given a molid and an experiment coordinate, retrieves the matrix of cooccurrences for the folds when the source of
    that particular molid was in test
    """
    MRDK = ManysourcesDataset(dset).mols()
    cooc, sources, _, _ = sources_coocurrences_df(dset=dset, feats=feats, model=model, lso=lso)
    source_of_molid = MRDK.molid2source(molid)
    index_of_source = np.where(sources == source_of_molid)[0][0]
    col = cooc[:, index_of_source]  # the column on which we put the condition
    interesting_Cooc = cooc[col]  # the matrix X
    # filter out the rows where we had troubles validating the model
    X = interesting_Cooc[expids, :]
    X = np.array(X, dtype=np.int)
    return X


def build_and_validate_regression_model(X, y, cv=10, model_factory=LinearRegression):
    # Cross-validation
    cver = cv_splits(len(y), y, num_folds=cv, stratify=False)
    r2_results = []
    for fold in range(cv):
        model = model_factory()
        train_indices, test_indices = cver(fold)
        Xtrain = X[train_indices, :]
        ytrain = y[train_indices]
        Xtest = X[test_indices, :]
        ytest = y[test_indices]
        # Train
        model.fit(Xtrain, ytrain)
        # Check results
        yhat = model.predict(Xtest)
        r2_results.append(r2_score(ytest, yhat))
    # Build full model
    model = model_factory()
    model.fit(X, y)
    weights = model.coef_

    return np.mean(np.array(r2_results)), weights, model


def generate_df_results(molid, importances, dset, feats, model, calibration, lso):
    cooccurrences, molids, expids, folds = molecules_coocurrences_df(dset=dset, feats=feats, model=model, lso=lso)
    df_losses, folds_df = read_losses(dset=dset,
                                      feats=feats,
                                      model=model,
                                      calibration=calibration,
                                      lso=lso,
                                      also_folds=True)
    dico_for_df = defaultdict(dict)

    MRDK = ManysourcesDataset(dset).mols()
    for other_molid in molids:
        if other_molid == molid:
            continue
        dico_for_df[other_molid] = {'importance': importances[np.where(molids == other_molid)[0][0]],
                                    'cooc_loss': average_loss(molid,
                                                              other_molid,
                                                              cooccurrences,
                                                              molids,
                                                              expids,
                                                              df_losses),
                                    'smiles': MolToSmiles(MRDK.molid2mol(other_molid))}
    df = pd.DataFrame.from_dict(dico_for_df, orient='index')
    df.index.names = ['molid']
    df['relabsimportance'] = df.importance.abs() / df.importance.abs().sum()
    return df[['relabsimportance', 'importance', 'smiles', 'cooc_loss']]


def generate_df_results_source(molid, importances, dset, feats, model, calibration, lso):
    cooccurrences, sources, expids, folds = sources_coocurrences_df(dset=dset, feats=feats, model=model, lso=lso)
    df_losses, folds_df = read_losses(dset=dset,
                                      feats=feats,
                                      model=model,
                                      calibration=calibration,
                                      lso=lso,
                                      also_folds=True)
    dico_for_df = defaultdict(dict)

    MRDK = ManysourcesDataset(dset).mols()
    for src in sources:
        if src == MRDK.molid2source(molid):
            continue
        dico_for_df[src] = {'importance': importances[np.where(sources == src)[0][0]],
                            'cooc_loss': average_loss_source(molid, src, cooccurrences, sources, expids, df_losses,
                                                             dset)}
    df = pd.DataFrame.from_dict(dico_for_df, orient='index')
    df.index.names = ['source']
    df['relabsimportance'] = df.importance.abs() / df.importance.abs().sum()
    return df[['relabsimportance', 'importance', 'cooc_loss']]


def average_loss(molid1, molid2, mcoocs, molids, expids, losses_df):
    """
    Molid1 is the target molid (the one for which we have built the model)
    molid2 is the mol for which we know its importance for the molid1
    We want to compute the average loss of molid1 when molid2 is in test
    """
    # molecules index  # FIXME: this should be taken from hub / molecules
    molid1_index = np.where(molids == molid1)[0][0]
    molid2_index = np.where(molids == molid2)[0][0]
    # coocurrences
    target_in_test = mcoocs[:, molid1_index]
    impmol_in_train = ~mcoocs[:, molid2_index]  # FIXME: this must be parameterizable
    expids = expids[target_in_test & impmol_in_train]
    losses_mol1 = losses_df.loc[expids, molid1]
    return losses_mol1.mean()


def average_loss_source(molid, source, scoocs, sources, expids, losses_df, dset):
    """
    Molid is the target molid (the one for which we have built the model)
    source2 is the source for which we know its importance for the molid1
    We want to compute the average loss of molid1 when source2 is in test
    """
    MRDK = ManysourcesDataset(dset).mols()
    source1 = MRDK.molid2source(molid)
    # sources index  # FIXME: this should be taken from hub / molecules
    source1_index = np.where(sources == source1)[0][0]
    molids_in_source = MRDK.source2molids(source)
    source2_index = np.where(sources == source)[0][0]
    # coocurrences
    target_in_test = scoocs[:, source1_index]
    impsrc_in_train = ~scoocs[:, source2_index]  # FIXME: this must be parameterizable
    expids = expids[target_in_test & impsrc_in_train]
    losses_mol1 = losses_df.loc[expids, molids_in_source]
    return losses_mol1.mean().mean()


def do_for_one_molid(calibration, dset, feats, lso, model, molid, results_dir, rm_factory, by_source=False):
    print molid
    MRDK = ManysourcesDataset(dset).mols()  # FIXME: this is read on each job, so once per molecule ATM...
    # Train and evaluate the model
    y, expids = get_y(molid, dset, feats, model, calibration, lso)
    if not by_source:
        X = get_X(molid, expids, dset, feats, model, lso)
    else:
        X = get_X_source(molid, expids, dset, feats, model) # makes no sense to run by source on LSO=False
    X = ~X  # coocurrences in train, less sparse, but better interpretation unless we tweak well the numbers...
    rsquared, feat_weights, trained_model = build_and_validate_regression_model(X, y, model_factory=rm_factory)
    rsquared = float(rsquared)
    # REMOVE moldir shows r2
    moldir = op.join(results_dir, 'r2=%.2f__%s' % (rsquared, molid))
    ensure_dir(moldir)
    # Save the model
    pd.to_pickle(trained_model, op.join(moldir, 'model_trained_rsquare=%.2f.pkl' % rsquared))
    # Save the smiles
    smiles = MolToSmiles(MRDK.molid2mol(molid))
    with open(op.join(moldir, 'smiles.txt'), 'w') as writer:
        writer.write(smiles)
    # Save the molecule-influence table
    if not by_source:
        df = generate_df_results(molid, feat_weights, dset, feats, model, calibration, lso)
        pd.to_pickle(df, op.join(moldir, 'results_df.pkl'))
        df.loc[molid] = (1E16, rsquared, smiles, np.mean(y))  # FIXME
        df['label'] = map(MRDK.molid2label, df.index)
        df = df[['label', 'relabsimportance', 'importance', 'smiles', 'cooc_loss']]
        df = df.sort('relabsimportance', ascending=False)
        df.head(20).to_html(op.join(moldir, 'results_df.html'))
    else:
        df = generate_df_results_source(molid, feat_weights, dset, feats, model, calibration, lso)
        pd.to_pickle(df, op.join(moldir, 'results_df_bysource.pkl'))
        df = df.sort('relabsimportance', ascending=False)
        df.head(20).to_html(op.join(moldir, 'results_df_bysource.html'))
    # Plot the distribution of losses (y)
    plt.figure()
    seaborn.distplot(y, bins=40)
    plt.xlim((-0.05, 1.05))
    plt.title('molid=%s, r2=%.2f' % (molid, rsquared))
    plt.savefig(op.join(moldir, 'y_dist.png'), bbox_inches='tight')
    plt.close()
    # --- WIP gridspec with chemdeco pics and things like that
    if not by_source:
        show_top = 4
        gs = gridspec.GridSpec(show_top, 2)
        fig = plt.figure(figsize=(24, 16))
        # Plot the molecule itself
        ax_mol = fig.add_subplot(gs[0:show_top / 2, 0])
        ax_mol.grid(False)
        ax_mol.get_xaxis().set_ticks([])
        ax_mol.get_yaxis().set_ticks([])
        mol = MRDK.molid2mol(molid)
        AllChem.Compute2DCoords(mol)
        ax_mol.imshow(artdeco2(rdkit2im(mol, size=(400, 400)), color='red' if df.loc[molid]['label'] == 'INHIBITOR' else 'green', chorrada=5))
        # Plot the distribution of losses
        ax_distr = fig.add_subplot(gs[show_top / 2:0, 0])
        seaborn.distplot(y, bins=40, ax=ax_distr)
        # Plot the top (we should align all to a common scaffold and maybe highlight substructures that matter)
        for rank, (inf_molid, row) in enumerate(df.iloc[1:show_top + 1].iterrows()):
            ax_influential_mol = fig.add_subplot(gs[rank, 1])
            ax_influential_mol.grid(False)
            ax_influential_mol.get_xaxis().set_ticks([])
            ax_influential_mol.get_yaxis().set_ticks([])
            mol_color = 'red' if row['label'] == 'INHIBITOR' else 'green'
            good_or_bad_color = 'red' if row['importance'] > 0 else 'green'
            # add decos
            mol = MRDK.molid2mol(inf_molid)
            AllChem.Compute2DCoords(mol)
            image = rdkit2im(mol)
            image = artdeco1(image, decos=(('black', good_or_bad_color),))
            image = artdeco2(image, color=mol_color)
            ax_influential_mol.imshow(image)
            ax_influential_mol.set_title('%s, inf=%.4f, cooc_loss=%.4f' %
                                         (inf_molid, row['importance'], row['cooc_loss']))
            # FIXME: cooc_loss also with stddev and standard error

        fig.suptitle('%s, r2=%.2f, cooc_loss=%.4f +/- %.4f' % (molid, rsquared, float(np.mean(y)), float(np.std(y))))
        plt.savefig(op.join(moldir, 'verde_que_te_quiero_verde.png'), bbox_inches='tight')
        plt.close()


def do_the_job(dset,
               feats,
               model,
               calibration=None,
               lso=True,
               regression_model=('linreg', LinearRegression),
               results_dir=op.join(MANYSOURCES_DATA_ROOT, 'results', 'loss_by_cooc'),
               n_jobs=None,
               by_source=False):
    rm_name, rm_factory = regression_model

    results_dir = op.join(results_dir,
                          'dset=%s' % dset,
                          'feats=%s' % feats,
                          'model=%s' % model,
                          'calibration=%s' % calibration,
                          'LSO=%r' % lso,
                          'reg_model=%s' % rm_name,
                          'bysource=%r' %by_source)
    ensure_dir(results_dir)

    _, molids, _, _ = molecules_coocurrences_df(dset=dset, feats=feats, model=model, lso=lso)

    if n_jobs is None:
        n_jobs = cpu_count()

    Parallel(n_jobs=n_jobs)(delayed(do_for_one_molid)(calibration,
                                                      dset, feats, lso, model,
                                                      molid, results_dir, rm_factory, by_source)
                            for molid in sorted(molids))


if __name__ == '__main__':
    do_the_job('hERG', 'ecfps1', 'logreg3', lso=True,
               regression_model=('ridge__alpha=0.001', partial(Ridge, alpha=1E-3)),
               n_jobs=None, by_source=True)
