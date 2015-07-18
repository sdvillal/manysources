# coding=utf-8
import seaborn as sns
import matplotlib.pyplot as plt

from manysources.hub import Hub, DEFAULT_EXPIDS


def brier_scores(dset_id='bcrp',
                 expids=DEFAULT_EXPIDS,
                 model='logreg3',
                 feats='ecfps1',
                 score_norm='0-1'):
    plt.figure()
    hub_lso, hub_crs = Hub.lchubs(dset_id=dset_id, expids=expids, model=model, feats=feats, score_norm=score_norm)
    lso_losses = hub_lso.squared_losses().mean(axis=1)
    crs_losses = hub_crs.squared_losses().mean(axis=1)
    sns.violinplot(vals=(lso_losses.values, crs_losses.values), names=('lso', 'crs'))
    plt.ylim(0, 1)
    plt.title(score_norm)


def aucs(dset_id='bcrp',
         expids=DEFAULT_EXPIDS,
         model='logreg3',
         feats='ecfps1',
         score_norm='0-1'):
    plt.figure()
    hub_lso, hub_crs = Hub.lchubs(dset_id=dset_id, expids=expids, model=model, feats=feats, score_norm=score_norm)
    lso_losses = hub_lso.squared_losses().mean(axis=1)
    crs_losses = hub_crs.squared_losses().mean(axis=1)
    sns.violinplot(vals=(lso_losses.values, crs_losses.values), names=('lso', 'crs'))
    plt.ylim(0, 1)
    plt.title(score_norm)

if __name__ == '__main__':
    brier_scores(score_norm=None)
    brier_scores(score_norm='0-1')
    brier_scores(score_norm='all')
    brier_scores(score_norm='max-cheating')
    brier_scores(score_norm='other-folds')
    brier_scores(score_norm='std-log')
    plt.show()


