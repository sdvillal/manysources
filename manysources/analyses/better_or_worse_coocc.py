'''
For each experiment: are the resulting losses (or AUC) better or worse than average?
'''
from manysources.analyses.losses import read_losses
from collections import defaultdict
import pandas as pd
import numpy as np


def average_loss(dset, feats, model, lso, calibration):
    """
    At each expid, we get 1 loss per molecule. We average this. Then we average across all expids
    Returns: a dataframe with one row per expid,foldid and the average loss as a column
             the average of all losses across all expids
             the standard deviation of all losses across all expids
    """
    df_losses, df_folds = read_losses(dset=dset, feats=feats, model=model, calibration=calibration, lso=lso,
                                      also_folds=True)
    #print df_folds
    df_mean = df_losses.mean(axis=1)   # for each expid, we get the average loss across all molecules
    total_mean = df_mean.mean(axis=0)
    total_std = df_mean.std(axis=0)
    # Now what we really want is how each split performed, so we need to add a mask at each expid depending on the fold
    # so as to get a expid,foldid: mean loss for each fold of each expid
    df_means_by_fold = [df_losses[df_folds==fold].mean(axis=1) for fold in range(10)]
    big_df = pd.concat(df_means_by_fold, axis=1)  # df with expids rows and 10 columns (max 10 folds per expids)
    big_df['expid'] = big_df.index   # copy index so it does not get lost during melting
    tidy_df = pd.melt(big_df, value_name='mean loss', var_name='foldid', id_vars='expid').dropna()

    return tidy_df, total_mean, total_std


def to_y(tidy_df, mean, std):
    # We could choose to apply many rules. For now: if the loss is higher than mean+std --> class 1, if loss is lower
    # than mean-std --> class 0. Drop the rest
    tidy_df['class1'] = tidy_df['mean loss'] > (mean + std)
    tidy_df['class0'] = tidy_df['mean loss'] < (mean - std)
    # drop the lines where we are not in class 1 nor class 2
    tidy_df = tidy_df[(tidy_df.class0 != False) | (tidy_df.class1 != False)] # at least one of the 2 columns is True
    bla = tidy_df.drop('class0', axis=1)
    y_df = bla.drop('mean loss', axis=1)
    y_df['class1'] = y_df['class1'].astype('int')
    return y_df


def get_xy(dset, feats, model, lso, y_df):
    from manysources.analyses.cooccurrences import molecules_coocurrences_df
    # get cooccurrences of compounds, along with the corresponding expids and fold ids as lists
    cooc, molids, expids, folds = molecules_coocurrences_df(dset, feats=feats, model=model, lso=lso)
    #print coocurrences
    #print expids
    #print folds
    cooccurrences_dict = defaultdict(list)
    for i in range(len(cooc)):
        cooccurrences_dict[(expids[i], folds[i])] = cooc[i]

    expids_in_y = y_df['expid']
    folds_in_y = y_df['foldid']
    y = np.array(y_df['class1'])
    X = []
    [X.append(cooccurrences_dict[(expid,foldid)]) for expid,foldid in zip(expids_in_y, folds_in_y)]
    X = np.array(X, dtype=np.int)
    return X, y


def cv_splits(num_points, Y, num_folds, rng=None, stratify=True, banned_train=None, banned_test=None):
    """
    (Stratified) cross-validation (from oscail.common.evaluation).

    Parameters:
      - num_points: the number of elements to split
      - Y: the group for each point (e.g. the class, the score, the source...)
      - num_folds: the number of splits
      - rng: an instance of a python/numpy random number generator
      - stratify: if True, a best effort is carried to keep consistent the Y proportions in each split
      - banned_train: a list of indices to not include in train (e.g. non-target class for OCC) or None
      - banned_test: a list of indices to not include in test or None

    Returns a function that maps a fold index (from 0 to num_folds-1) to the indices of its train/test instances.
    """
    if rng is None:
        rng = np.random.RandomState(0)
    permutation = rng.permutation(num_points)
    if stratify:
        permutation = permutation[np.argsort(Y[permutation])]
    folds = [permutation[base::num_folds] for base in range(num_folds)]
    seed = rng.randint(1024*1024*1024)
    banned_train = set() if not banned_train else set(banned_train)
    banned_test = set() if not banned_test else set(banned_test)

    def cver(fold):
        if fold >= num_folds:
            raise Exception('There are not so many folds (requested fold %d for a cross-val of %d folds)' %
                            (fold, num_folds))
        rng = np.random.RandomState(seed)
        # Now select and permute test indices, removing the banned items.
        # N.B. Order of operations matter...
        test_indices = folds[fold][rng.permutation(len(folds[fold]))]
        ban_in_train = set(test_indices) | banned_train
        train_indices = [train_i for train_i in rng.permutation(num_points)
                         if train_i not in ban_in_train]
        test_indices = [test_i for test_i in test_indices if test_i not in banned_test]
        return train_indices, test_indices

    return cver



def build_and_validate_logreg(X, y, cv=10):
    from sklearn.linear_model import LogisticRegression
    from sklearn.grid_search import GridSearchCV
    from sklearn.metrics import roc_auc_score
    parameters = {'penalty': ['l1', 'l2'], 'C': [1E-6, 1E-5, 1E-4, 1E-3, 1E-2, 1E-1, 0.5, 1, 1.5, 5, 10, 100]}
    model = LogisticRegression()
    model = GridSearchCV(model, parameters, n_jobs=4)

    cver = cv_splits(len(y), y, num_folds=cv)
    auc_results = []
    for fold in range(cv):
        print fold
        print y
        train_indices, test_indices = cver(fold)
        Xtrain = X[train_indices, :]
        ytrain = y[train_indices]
        Xtest = X[test_indices, :]
        ytest = X[test_indices, :]
        # Train
        model.fit(Xtrain, ytrain)
        # Check results
        print model.best_score_
        print model.best_params_
        yhat = model.predict_proba(Xtest)
        auc_results.append(roc_auc_score(ytest, yhat))

    return np.mean(np.array(auc_results))


def build_logreg(X, y, penalty, C):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty=penalty, C=C)
    m = model.fit(X,y)
    coeffs = m.coef_
    print len(coeffs[0])
    print coeffs



if __name__ == '__main__':
    tidy_df, total_mean, total_std = average_loss('bcrp', 'ecfps1', 'logreg3', True, '0-1')
    y_df = to_y(tidy_df, total_mean, total_std)
    X, y = get_xy('bcrp', 'ecfps1', 'logreg3', True, y_df)
    build_logreg(X, y, penalty='l2', C=0.001)