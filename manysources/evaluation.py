# coding=utf-8
import numpy as np
from manysources.common.evaluation import GroupsSplitCV, JaggedCV
from manysources.datasets import ManysourcesDataset


############################
# LSO Cross-Validation
############################

def setup_splitters_lsocv(dset=ManysourcesDataset('bcrp'),
                          molids=None,
                          expids=None):
    """
    Generates grouped cross-validation splitters for the molecules in the dataset.

    Parameters
    ----------
    dset: ManysourcesDataset
        The dataset we will split on

    molids: string list, default None
        The molids to consider
        if None, all molecules in the dataset are considered

    expids: int list, default None
        The experiment ids; these are integers that identyify a single experiment and are used to seed rng
        If None, around 120 expids are generated (40 * number of fold sizes proposed by dset.lso_numcvfolds())


    Returns
    -------
    A list of GroupsSplitCV splitters, where the sources of the molecules are taken as groups and the expid
    for each individual splitter can be retrieved as "splitter.seed".

    Examples
    --------
    In the bcrp dataset there are 47 sources and 978 compounds.
    >>> dset = ManysourcesDataset('bcrp')
    >>> expids = (3, 7, 31, 127, 8191, 131071, 524287, 2147483647)
    >>> splitters = setup_splitters_lsocv(dset, expids=expids)
    >>> len(splitters) == len(expids)
    True
    >>> splitters[7].seed
    2147483647
    >>> same_splitters = setup_splitters_lsocv(dset, expids=(2147483647,))
    >>> same_splitters[0].seed
    2147483647
    >>> same_splitters[0].num_folds == splitters[7].num_folds
    True
    >>> from itertools import chain
    >>> trest_sets_one = list(chain(*[splitter.split()[1] for splitter in splitters[7]]))
    >>> trest_sets_two = list(chain(*[splitter.split()[1] for splitter in same_splitters[0]]))
    >>> trest_sets_one == trest_sets_two
    True
    >>> len(trest_sets_one) == len(dset.mols().molids())
    True
    >>> len(set(trest_sets_one)) == len(dset.mols().molids())
    True
    """
    molids = molids if molids is not None else dset.mols().molids()
    if expids is None:
        expids = range(40 * len(dset.lso_numcvfolds()))
    lsocvs = []
    for expid in expids:
        num_folds = np.random.RandomState(expid).choice(dset.lso_numcvfolds())
        # We setup the grouped-cv splitters
        lsocv = GroupsSplitCV(dset.mols().molids2sources(molids=molids),
                              seed=expid,
                              num_folds=num_folds)
        lsocvs.append(lsocv)
    return lsocvs


def crscv_from_lsocv(lsocv, seed=None):
    """
    Returns a "JaggedCV", "Classical Random Splitting", surrogate to the lsocv object.
    The returned splitters will provide the same split sizes as lsocv.
    If seed is None, the same seed as lsocv is also used.
    """
    if seed is None:
        seed = lsocv.seed
    return JaggedCV(seed, [len(splitter.split()[1]) for splitter in lsocv.splitters()])