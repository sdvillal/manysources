# coding=utf-8
from itertools import product

from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor
from sklearn.linear_model import Ridge
import numpy as np
from sklearn.linear_model.base import LinearRegression

from manysources.hub import Hub


def regress_the_loss_from_coocurrences_Xy(hub, molid, as_float=False, reverse=False):
    """
    Returns X and y ready to regress the loss from the coocurrences matrix.

    Parameters
    ----------
    hub: The experiments hub
    molid: The molecule id to create X and y against
    as_float: Return X and y as float arrays (otherwise they are bool arrays)
    reverse: If True, X means "source/molecule in train"; else X means "source/molecule in train"

    Returns
    -------
    A tuple (X, y).
      - X is a num_experiments x (num_sources (if hub is lso) or num_molecules (if hub is crs)) of
      in test coocurrences.
      - y are the losses of the molecule in the experiment
    :rtype: (np.ndarray, np.ndarray)
    """
    # Dependent variable: molecule loss
    y = hub.squared_losses()[molid]
    # Independent variables: molecule coocurrences or source coocurrences
    X = hub.mcoocs() if not hub.lso else hub.scoocs()
    molid = molid if not hub.lso else hub.mols().molid2source(molid)
    # Folds in which the molecule was in test
    X = X[X[molid]]
    # Dissolve the multi-index, we now only have one fold per expid
    X = X.reset_index()
    X = X.set_index('expid', verify_integrity=True)
    # We get the problematic expids in X, so we need to select these which completed
    X = X.loc[y.index]
    # We do not need the fold, the molid (source) column is constant so irrelevant
    del X['fold']
    del X[molid]
    # Make "in_train" X, instead of "in_test"?
    if reverse:
        X = ~X
    # Everything I have tried digests well boolean matrices, but just in case...
    if as_float:
        X = X.astype(float)
        y = y.astype(float)
    # Done...
    return X, y


def regress_the_loss(hub, molid, regressor=Ridge()):
    """Example of regression to the loss from coocurrences."""

    X, y = regress_the_loss_from_coocurrences_Xy(hub, molid)

    # Fit
    regressor.fit(X, y)

    # Look at the model
    coeffs = regressor.coef_            # Per molecule coeff
    r2 = regressor.score(X, y)          # R^2
    scores = regressor.predict(X)       # Per experiment resubstitution score
    perexpid_loss = (y - scores) ** 2   # Per experiment resubstitution loss (squared error)

    # Most influential molecules
    larger = np.argsort(-np.abs(coeffs))[:20]

    return molid, zip(X.columns[larger], coeffs[larger]), r2, perexpid_loss


def rfr_the_loss(hub, molid):
    X, y = regress_the_loss_from_coocurrences_Xy(hub, molid)
    # rfr = RandomForestRegressor(n_estimators=800, n_jobs=8, oob_score=True, random_state=0)
    rfr = GradientBoostingRegressor(n_estimators=100)
    rfr.fit(X, y)
    # print rfr.oob_score_
    # print rfr.oob_improvement_
    influential = np.argsort(-rfr.feature_importances_)[:20]
    print '\t%s' % '\n\t'.join(X.columns[influential])


if __name__ == '__main__':

    MOLIDS = [
        's=Matsson_2009__n=Bromosulfalein',
        's=Zembruski_2011__n=103268452',
        's=Patel_2011__n=19',
        's=Ochoa-Puentes_2011__n=131273183',
        's=Jin_2006__n=Ginsenoside Rg1',
        's=Matsson_2007__n=Timolol',
    ]

    hub_lso = Hub(dset_id='bcrp', lso=True, expids=range(40000))
    hub_csr = Hub(dset_id='bcrp', lso=False, expids=range(40000))

    for molid, hub in product(sorted(hub_csr.mols().molids()), (hub_lso, hub_csr)):
        # print molid, hub.lso
        # rfr_the_loss(hub, molid)
        molid, most_influential, r2, _ = regress_the_loss(hub, molid, regressor=LinearRegression())
        print molid, hub.lso, r2
        for infmolid, coeff in most_influential:
            print '\t %.4f %s' % (coeff, infmolid)
        print '-' * 80


# MOLID = 'CHEMBL1951453'            # hERG
# MOLID = 'NOCAS_M43'                # mutagenicity, FAILS with BAD SMELL
# MOLID = '74-83-9'                  # mutagenicity
# MOLID = 'Bromocriptine'            # pgp-cruciani, BSEP HIT!!!
# MOLID = 'Succinylsulfathiazole'    # pgp-cruciani

# Could we also use the fact that ridge implements multi-variate regression?
# No, seems unlikely, as each molecule X is unique (but for same-source, lso)

# Maybe just a vanilla linearregression could do the trick
# We could use trees too
# We could use also SVR and the like, but would need to use different interpretation
#   - support vectors would be important experiments
#   - if we use linear kernels we could go back to individual mols
#   - maybe we can go back regardless of the kernel (as the design matrix only contains 1s and 0s)

# What about converting to (binary/multiclass) classification problems?
# Would need to cleverlly discretize Y to, e.g. {"good_classifier", "bad_classifier"}

# What about higher order interactions?
# We would need to add "combinations of 2", "combinations of 3"... features
# Vowpal Wabbit is our friend there
