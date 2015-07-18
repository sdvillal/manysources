# coding=utf-8
"""Link together information on examples, models and evaluation for easier queries."""
from itertools import izip
import re
from numba import jit
import os.path as op

import numpy as np
import pandas as pd
from pandas.core.index import MultiIndex
from sklearn.decomposition import NMF
from whatami import whatable

from manysources.analyses.cooccurrences import molecules_coocurrences_df, sources_coocurrences_df
from manysources.analyses.logreg_weights import logreg_weights
from manysources.analyses.losses import read_losses
from manysources.common.misc import ensure_dir
from manysources.datasets import ManysourcesDataset, MANYSOURCES_DATA_ROOT
from manysources.experiments import DEFAULT_EXPIDS, MANYSOURCES_FEATS
from pillow_utils import rdkit2im
from manysources.analyses.scores import merge_cache_scores


@whatable
class Hub(object):

    def __init__(self,
                 dset_id='bcrp',
                 expids=DEFAULT_EXPIDS,
                 lso=True,
                 model='logreg3',
                 feats='ecfps1',
                 score_norm=None):
        super(Hub, self).__init__()
        self.dset_id = dset_id
        self.lso = lso
        self.model = model
        self.feats = feats
        self.score_norm = score_norm
        self.expids = expids
        # lazy
        self._dset = None
        self._df_losses = None

    _HUBS = {}

    @classmethod
    def hub(cls,
            dset_id='bcrp',
            expids=tuple(range(4096)),
            lso=True,
            model='logreg3',
            feats='ecfps1',
            score_norm=None):
        hub = cls(dset_id=dset_id, expids=expids, lso=lso, model=model, feats=feats, score_norm=score_norm)
        hid = hub.what().id()
        if hid not in Hub._HUBS:
            Hub._HUBS[hid] = hub
        return Hub._HUBS[hid]

    @classmethod
    def lchubs(cls,
               dset_id='bcrp',
               expids=tuple(range(4096)),
               model='logreg3',
               feats='ecfps1',
               score_norm=None):
        return cls.hub(dset_id=dset_id, expids=expids, model=model, feats=feats, score_norm=score_norm, lso=True), \
            cls.hub(dset_id=dset_id, expids=expids, model=model, feats=feats, score_norm=score_norm, lso=False)

    def dset(self):
        """
        Returns the dataset hub.

        :rtype: ManysourcesDataset
        """
        if self._dset is None:
            self._dset = ManysourcesDataset(dset=self.dset_id)
        return self._dset

    def mols(self):
        """
        Returns the molecules side of life of the dataset hub.

        :rtype: manysources.common.molecules.MoleculesFromRDKit
        """
        return self.dset().mols()

    def aucs(self):
        """
        Returns a dataframe with as many rows as experiments and three columns: (mean AUC, stddev AUC, stacked AUC).

        It would look like this:

        |----------------------------------------------------|
        |        | meanROCAUC | stddevROCAUC | stackedROCAUC |
        |----------------------------------------------------|
        | expid1 |     0.75   |     0.04     |      0.76     |
        | expid2 |     0.78   |     0.04     |      0.76     |
        |  ...   |     ...    |     ...      |      ...      |
        |----------------------------------------------------|

        :rtype: pandas.DataFrame
        """
        raise NotImplementedError()

    def scores(self, tidy=True):
        """If tidy, returns a dataframe looking like that:

        |---------------------------------|
        | molid  | expid | fold |  score  |
        |---------------------------------|
        | molid1 |   1   |   3  | 0.2334  |
        | molid1 |   2   |   0  | 0.4553  |
        |  ...   |  ...  |  ... |   ...   |
        |---------------------------------|

        Else, returns a dataframe with expid as index and as many columns as mols (in which case we will
        take into account self.lso).

        |--------------------------------|
        |        |  mol1  |  mol2  | ... |
        |--------------------------------|
        | expid1 | 0.2334 |  0.783 | ... |
        | expid2 | 0.2054 |  0.541 | ... |
        |  ...   |  ...   |   ...  | ... |
        |--------------------------------|
        """

        scores, expids, folds, molids, y = merge_cache_scores(dset_id=self.dset_id,
                                                              model=self.model,
                                                              feats=self.feats,
                                                              calib=self.score_norm,
                                                              lso=self.lso)

        df = pd.DataFrame(data=scores.T, index=expids, columns=molids)

        # select the relevant expid only
        # TODO: allow to read only these in scores_df
        # FIXME: warn if we cannot find all requested expids, document that can be the case
        expids = [expid for expid in self.expids if expid in expids]
        df = df.loc[expids]

        if not tidy:
            return df
        else:
            df['expid'] = df.index  # use reset_index
            tidy_df = pd.melt(df, id_vars='expid', var_name='molid', value_name='score')
            folds_tidy = np.concatenate(folds)
            tidy_df['fold'] = folds_tidy
            return tidy_df[['molid', 'expid', 'fold', 'score']]

    def squared_losses(self):
        """
        Returns a dataframe with squared losses per experiment and molecule (can be used to compute, e.g., Brier score).

        Returns a pandas dataframe (expid, molid) -> loss.
        Rows and cols appear sorted.
        Loss is squared loss: (label - score) ** 2
        NaNs (failed experiments) are removed.


        It would look like this:

        |----------------------------|
        |        | mol1 | mol2 | ... |
        |----------------------------|
        | expid1 | 0.75 | 0.23 | ... |
        | expid2 | 0.31 | 0.33 | ... |
        |  ...   |  ... | ...  | ... |
        |----------------------------|

        :rtype: pandas.DataFrame
        """
        if self._df_losses is None:
            dfl, _ = read_losses(dset=self.dset_id,
                                 expids=self.expids,
                                 feats=self.feats,
                                 model=self.model,
                                 lso=self.lso,
                                 calibration=self.score_norm)
            self._df_losses = dfl.sort_index(axis=0).sort_index(axis=1).dropna(axis=1)
        return self._df_losses

    def mcoocs(self):
        """
        Returns a multilevel-indexed dataframe of molecules coocurrences in test for each partition train/test.

        The dataframe returned by this function has
          - a sorted index with two levels (expid, fold)
          - a sorted column index, one column per molecule
          - boolean values
        It would look like this

        |----------------------------------------|
        |     index       |        data          |
        |-----------------|----------------------|
        | expid  | foldid |  mol1  | mol2  | ... |
        |----------------------------------------|
        |   0    |   0    |  False | False | ... |
        |   1    |   0    |  True  | False | ... |
        | ...    |  ...   |  ...   |  ...  | ... |
        |----------------------------------------|

        :rtype: pandas.DataFrame
        """
        mcoocs, molids, expids, folds = molecules_coocurrences_df(
            expids=self.expids,
            dset=self.dset_id,
            feats=self.feats,
            model=self.model,
            lso=self.lso
        )
        index = MultiIndex.from_arrays(arrays=(expids, folds))
        index.levels[0].name = 'expid'
        index.levels[1].name = 'fold'
        mcooc_df = pd.DataFrame(data=mcoocs,
                                index=index,
                                columns=molids)

        return mcooc_df.sort_index(axis=0).sort_index(axis=1)

    def scoocs(self):
        """
        Returns a multilevel-indexed dataframe of sources coocurrences in test for each partition train/test.

        The dataframe from this palyndromic function has:
          - a sorted index with two levels (expid, fold)
          - a sorted column index, one column per source
          - boolean values
        It would look like this

        |----------------------------------------|
        |     index       |        data          |
        |-----------------|----------------------|
        | expid  | foldid |  src1  | src2  | ... |
        |----------------------------------------|
        |   0    |   0    |  False | False | ... |
        |   1    |   0    |  True  | False | ... |
        | ...    |  ...   |  ...   |  ...  | ... |
        |----------------------------------------|

        :rtype: pandas.DataFrame
        """
        scoocs, sources, expids, folds = sources_coocurrences_df(
            expids=self.expids,
            dset=self.dset_id,
            feats=self.feats,
            model=self.model,
            lso=self.lso
        )
        index = MultiIndex.from_arrays(arrays=(expids, folds))
        index.levels[0].name = 'expid'
        index.levels[1].name = 'fold'
        mcooc_df = pd.DataFrame(data=scoocs,
                                index=index,
                                columns=sources)

        return mcooc_df.sort_index(axis=0).sort_index(axis=1)

    def mean_loss_matrix(self, mols_in_rows=True, rows_in_train=False):
        """
        Returns two dataframes: mean loss and number of occurrences.

        These dataframes:
          - have mols or sources in rows (depending of *mols_in_rows*)
          - have mols in columns
          - each entry contains the mean loss (and count) when
            * the mol in col is in test
            * whatever in the row is in train or test (depending on *rows_in_train*)

        They would look like this:

        |-----------------------------------------|
        |     index       |        data           |
        |-----------------|-----------------------|
        | source or molid | molid1 | molid2 | ... |
        |-----------------------------------------|
        |   molid1        |  0.83  |  0.02  | ... |
        |   molid2        |  0.17  |  0.01  | ... |
        | ...             |  ...   |  ...   | ... |
        |-----------------------------------------|

        |-----------------------------------------|
        |     index       |        data           |
        |-----------------|-----------------------|
        | source or molid | molid1 | molid2 | ... |
        |-----------------------------------------|
        |   molid1        |  794   |  733   | ... |
        |   molid2        |  680   |  667   | ... |
        | ...             |  ...   |  ...   | ... |
        |-----------------------------------------|

        :rtype: (pandas.DataFrame, pandas.DataFrame)
        """

        matid = 'mir=%r#rit=%r' % (mols_in_rows, rows_in_train)
        cache_dir = op.join(MANYSOURCES_DATA_ROOT, 'results', 'mlms', self.what().id(maxlength=1))
        ensure_dir(cache_dir)
        cache_file = op.join(cache_dir, matid + '.pkl')

        if not op.isfile(cache_file):

            # row masks
            rows_df = self.mcoocs() if mols_in_rows else self.scoocs()
            rows_df = ~rows_df if rows_in_train else rows_df
            # col masks
            cols_df = self.mcoocs()
            # losses
            loss_df = self.squared_losses().T

            # sanity checks for columns
            assert np.all(cols_df.columns == loss_df.index)

            @jit(nopython=True)
            def update_cooc_cooc(res_matrix, norm_matrix, losses, rows, cols):
                for row in rows:
                    for col in cols:
                        res_matrix[row, col] += losses[col]
                        norm_matrix[row, col] += 1

            loss_matrix = np.zeros((len(rows_df.columns), len(cols_df.columns)))
            norm_matrix = np.zeros((len(rows_df.columns), len(cols_df.columns)))

            for expid in loss_df.columns:
                print 'Computing mean loss for expid=%d' % expid
                if expid not in rows_df.index.levels[0]:
                    print '\t Cannot find coocurrences, skipping...'
                    continue
                exp_losses = loss_df[expid]
                #
                # FIXME: this does not seem right now that we have a multilevel df
                #
                for rowmask, colmask in izip(rows_df.loc[expid].values,
                                             cols_df.loc[expid].values):
                    update_cooc_cooc(loss_matrix,
                                     norm_matrix,
                                     exp_losses.values,
                                     np.where(rowmask)[0],
                                     np.where(colmask)[0])

            losses_df = pd.DataFrame(loss_matrix / norm_matrix,
                                     index=sorted(rows_df.columns),
                                     columns=cols_df.columns)

            counts_df = pd.DataFrame(norm_matrix,
                                     index=sorted(rows_df.columns),
                                     columns=cols_df.columns)

            pd.to_pickle((losses_df, counts_df), cache_file)

        return pd.read_pickle(cache_file)

    def fold_coocurrence_tensor(self):
        raise NotImplementedError('To implement and myabe even factorize with PARAFACT')

    def as_pil_images(self):
        # generates 2D PIL representations of the molecules
        # TODO: this must go to mols class
        return [rdkit2im(mol) for mol in self.mols()]

    def logreg_models(self):
        """
        Returns a 4-tuple with the weights for the logistic regressions, intercepts and experiment coordinates.

         - the first element is is a num_rows x num_features sparse matrix;
           each row corresponds to the weights for an "expid-fold" model
           the colums are the features

         - the second  element is a num_rows array of logistic regression intercepts

         - the third and fourth elements are num_rows long numpy arrays with the coordinates of the experiments;
           these are expids + fold_ids (e.g. (expid=2, fold=1))

        :rtype: (scipy.sparse.csr_matrix, np.array, np.array, np.array)
        """
        return logreg_weights(dset=self.dset_id,
                              expids=self.expids,
                              feats=self.feats,
                              model=self.model,
                              lso=self.lso)

    def molid2index(self, molid, molids=None):
        # FIXME: this is better from molecules and whatnot, inefficient, with little petits eyes pressure
        if molids is None:
            molids, _, _ = self.feature_matrix()
        return np.where(molids == molid)[0][0]

    def feature_matrix(self):
        """
        Returns the feature matrix, along with the relevant molids (these for which feature extraction worked)
        and the labels.

        The feature matrix can be either a dense numpy array or an sparse matrix (so no type annotation, sorry).
        """
        molids, X, y = MANYSOURCES_FEATS[self.feats].extractor(self.dset())
        return molids, X, y

    def mol_feats(self, molid):
        """Returns a numpy array with the indices of the features occurring in the molecule."""
        molids, X, y = self.feature_matrix()
        mol_index = self.molid2index(molid, molids=molids)
        if isinstance(X, np.ndarray):
            raise Exception('Probably this makes no sense, all is relevant, but we could just get the non-zero')
        return X[mol_index, :].indices

    def relevant_folds(self, molid):
        """Returns a list of tuples (expid, foldnum) where the molid was in test."""
        mcoocs = self.mcoocs()
        return mcoocs[mcoocs[molid]].index

    def relevant_models(self, molid):
        weights, intercepts, expids, foldnums = self.logreg_models()
        relevant_folds = self.relevant_folds(molid)
        rows = [row for row, (expid, foldnum) in enumerate(izip(expids, foldnums))
                if (expid, foldnum) in relevant_folds]
        return weights[rows, :], intercepts[rows], relevant_folds

    def smiles(self, findices):
        return self.dset().ecfps(no_dupes=True).i2s[findices]  # FIXME quick and dirty

    def logreg_replay(self, molid):
        # TODO: do we need to calibrate / normalise?
        #       in other words, should we replay all at the same time or read the scores from the already done preds?
        if 'logreg' not in self.model:
            raise Exception('logreg replay can only be applied to logistic regression models')
        relevant_feats = self.mol_feats(molid)
        smiles = self.smiles(relevant_feats)
        weights, intercepts, relevant_folds = self.relevant_models(molid)
        print weights.shape
        print relevant_feats
        print len(relevant_feats)
        relevant_weights = weights.tocsc()[:, relevant_feats].tocsr().toarray()  # FIXME to_dense
        nfeats = float(relevant_weights.shape[0] * relevant_weights.shape[1])
        # print relevant_weights.shape, relevant_weights.nnz / nfeats, len(relevant_feats)
        molas = relevant_weights.dot(np.ones(len(relevant_feats)))
        neg_feats = relevant_weights < 0
        pos_feats = relevant_weights > 0

        print neg_feats.sum(), pos_feats.sum()
        print relevant_weights.shape
        mean_weights = relevant_weights.mean(axis=0)
        order = mean_weights.argsort()
        print zip(smiles[order], ['%.4f' % num for num in mean_weights[order]])

        linear_term = np.exp(molas + intercepts)
        predictions = linear_term / (1 + linear_term)
        print predictions

        # + intercepts


def expid_only(names, keep_lso=False):
    """Given a list of names (e.g. the columns in losses), return a new list keeping only the expid#lso part."""
    if not keep_lso:
        return [name.partition('#lso=')[0] for name in names]
    p = re.compile(r'(expid=\d+#lso=(?:True|False))')
    return [p.search(name).group(0) for name in names]


def hubigraph(hub, edges='loss'):
    """Creates a nice ubigraph plot from a results hub.
    Nodes are molecules.
    Edges represent similarity in either:
      - logreg weight space
      - loss space
      - score space
    """

    mols = hub.mols().mols()

    if edges == 'loss':
        losses = hub.squared_losses()
        # print losses

        # X = losses.T.values
        # nns = NearestNeighbors()
        # nns.fit(X)
        # A = nns.kneighbors_graph(X, n_neighbors=5)
        # print A


def cooc_loss_matrix_expectations(hub):
    """Checks and documents expectations for the various coocurrence-loss matrices.
    The usefulness of such matrices is still to determine:
      - to find higher order interaction
    """

    # Kill the memory
    m2m_train, _ = hub.mean_loss_matrix(mols_in_rows=True, rows_in_train=True)
    m2m_trest, _ = hub.mean_loss_matrix(mols_in_rows=True, rows_in_train=False)
    s2m_train, _ = hub.mean_loss_matrix(mols_in_rows=False, rows_in_train=True)
    s2m_trest, _ = hub.mean_loss_matrix(mols_in_rows=False, rows_in_train=False)

    if hub.lso:
        # Check LSO expectations for the mean loss matrix

        # Each molecule in columns have the same score for all the molecules in the same source
        # The score is NaN for all the molecules in the same source
        # m2m_trest
        pass
    else:
        # Check CRS expectations for the mean loss matrix
        pass


if __name__ == '__main__':

    hub_lso = Hub(lso=True, dset_id='bcrp')
    hub_crs = Hub(lso=False, dset_id='bcrp')

    weights, intercepts, expids, foldids = hub_lso.logreg_models()

    # This seems correct too
    # Note that in LSO, we will have the same value for each molecule in the same source
    meanl, counts = hub_crs.mean_loss_matrix(mols_in_rows=True, rows_in_train=True)
    print meanl
    exit(22)

    nmf = NMF()
    print nmf.fit_transform(meanl)

    print('Done')
    exit(0)

#
# TODO: cache coocurrences if needed
#
