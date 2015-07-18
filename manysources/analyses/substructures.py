from collections import defaultdict
import os.path as op
from itertools import izip
import warnings
import cPickle as pickle
import glob
import math
import os

import h5py
import pandas as pd
import numpy as np
from rdkit import Chem
import matplotlib.pyplot as plt

from manysources import MANYSOURCES_ROOT
from manysources.datasets import ManysourcesDataset, MANYSOURCES_MOLECULES
from manysources.hub import Hub

warnings.simplefilter("error")

PROBLEMATIC_EXPIDS = {'bcrp':[489, 1840, 2705, 2780, 3842], 'hERG':[]}


def substructs_weights_one_source(source, model='logreg3', feats='ecfps1', dset='bcrp', num_expids=4096):
    """
    Given a source, what are the weights of all the substructures when this source is in train / in test for LSO for
    all requested expids. We now use the Hub. For the cases where the source is in train, it happens many times per
    expid so we take the average.
    """
    importances_source_in_lso = []
    expids = tuple(range(num_expids))
    hub = Hub(dset_id=dset, lso=True, model=model, feats=feats, expids=expids)
    source_coocs = hub.scoocs()

    # indices (expids, fold id) of source in test
    indices_in_test = source_coocs[source_coocs[source]].index
    indices_in_test = [(expid, foldnum) for (expid, foldnum) in indices_in_test if expid not in PROBLEMATIC_EXPIDS[dset]]

    # indices (expids, fold ids) of source in train
    indices_in_train = source_coocs[source_coocs[source]==False].index
    # transform it into a dictionary of {expids:[foldnums]}
    indices_in_train_dict = defaultdict(list)
    for expid, foldnum in indices_in_train:
        if expid not in PROBLEMATIC_EXPIDS[dset]:
            indices_in_train_dict[expid].append(foldnum)

    # get corresponding weights
    weights,_, expids, foldnums = hub.logreg_models()
    rows_out = [row for row, (expid, foldnum) in enumerate(izip(expids, foldnums))
                if (expid, foldnum) in indices_in_test]

    weights_in_test = weights[rows_out, :].todense()

    # For train, we get several foldnums per expids and we want to average those weights
    for expid_in in indices_in_train_dict.keys():
        rows = [row for row, (expid, fold) in enumerate(izip(expids, foldnums)) if expid == expid_in
                and fold in indices_in_train_dict[expid_in]]
        w = weights[rows, :]
        w = np.squeeze(np.asarray(w.tocsc().mean(axis=0)))
        importances_source_in_lso.append(w)

    return indices_in_train_dict.keys(), np.array(importances_source_in_lso), np.asarray(weights_in_test)


def proportion_relevant_features(source, dset='bcrp', model='logreg3', feats='ecfps1'):
    """
    Here, "relevant" refears to having a non null weight in the models.
    """
    sums_in = []
    sums_out = []
    expids, all_weights_in, all_weights_out = substructs_weights_one_source(source=source, model=model, feats=feats, dset=dset)
    for weights_in, weights_out in zip(all_weights_in, all_weights_out):
        sums_in.append(np.sum(weights_in != 0))
        sums_out.append(np.sum(weights_out != 0))
    return np.mean(np.array(sums_in)), np.mean(np.array(sums_out))


def most_changing_substructs_source(dset='bcrp', model='logreg3', feats='ecfps1', source='Imai_2004', top=10):
    """
    Returns a dictionary of {substruct:[changes in weight]} for all the expids in which the substructure was among the
    top most changing in terms of logistic weights (comparing weights when the source is in training or in test)
    """
    if not op.exists(op.join(MANYSOURCES_ROOT, 'data', 'results', dset, 'top_%i_substructures_changing_weight_%s_%s_%s.dict'
            %(top, source, model, feats))):
        substruct_changes_dict = defaultdict(list)
        expids, all_weights_in, all_weights_out = substructs_weights_one_source(source=source, model=model, feats=feats, dset=dset)
        i2s = ManysourcesDataset(dset).ecfps(no_dupes=True).i2s
        # dimensions: expids * num substructures
        # get the absolute weight difference between source in and source out
        difference_weights = np.absolute(np.subtract(all_weights_in, all_weights_out))
        orders = np.argsort(difference_weights, axis=1) # for each expid, indices of the sorted weight differences
        # Let's take top n differences
        for expid, o_i in enumerate(orders[:,-top:]):  # because the argsort puts first the smallest weight differences!
            great_substructs = [i2s[i] for i in o_i]
            corresponding_weights = difference_weights[expid][o_i]
            for i, sub in enumerate(great_substructs):
                substruct_changes_dict[sub].append(corresponding_weights[i])
        # substruct_changes_dict now contains the changes of weight obtained for all the expid in which the substruct was
        # among the top n changing substructs.
        with open(op.join(MANYSOURCES_ROOT, 'data', 'results', dset, 'top_%i_substructures_changing_weight_%s_%s_%s.dict'
                %(top, source, model, feats)), 'wb') as writer:
            pickle.dump(substruct_changes_dict, writer, protocol=pickle.HIGHEST_PROTOCOL)
        return substruct_changes_dict
    else:
        with open(op.join(MANYSOURCES_ROOT, 'data', 'results', dset, 'top_%i_substructures_changing_weight_%s_%s_%s.dict'
                %(top, source, model, feats)), 'rb') as reader:
            return pickle.load(reader)


def substructures_change_weight(source, model='logreg3', feats='ecfps1', dset='bcrp', non_zero_diff=0.01):
    """
    Given a source, retrieve substructures present in the source that on averagenchange weight when this source is in
    train / in test in LSO. Also returns the occurrences of these substructures in the 2 classes (inhibitior /
    non inhibitor)
    """
    _, weights_ins, weights_outs = substructs_weights_one_source(source, model=model, feats=feats, dset=dset)
    # average over all expids:
    weights_in = np.array(weights_ins).mean(axis=0)
    # average over all expids
    weights_out = np.array(weights_outs).mean(axis=0)
    i2s = ManysourcesDataset(dset).ecfps(no_dupes=True).i2s

    difference_weights = np.array(weights_in - weights_out)
    order = np.argsort(difference_weights)
    ordered_diff_w = difference_weights[order]
    ordered_substr = i2s[order]
    print '%i substructures have their weights decreased when the source %s is in external set (LSO)' \
          %(len(ordered_substr[ordered_diff_w > non_zero_diff]), source)
    print '%i substructures have their weights increased when the source %s is in external set (LSO)' \
          %(len(ordered_substr[ordered_diff_w < - non_zero_diff]), source)
    # Retrieve occurrence in source for each of those substructures with non-zero difference
    subs_dict = defaultdict(list)
    # 1. Decreased weight
    for subs, weight_diff in zip(ordered_substr[ordered_diff_w > non_zero_diff], ordered_diff_w[ordered_diff_w > non_zero_diff]):
        n1, n2 = substructure_appears_in_source(subs, source, dataset_nickname=dset)
        if (n1, n2) != (0,0):
            subs_dict[subs].append((n1, n2))
            subs_dict[subs].append(weight_diff)

    # 2. Increased weight
    for subs, weight_diff in zip(ordered_substr[ordered_diff_w < -non_zero_diff], ordered_diff_w[ordered_diff_w < - non_zero_diff]):
        n1, n2 = substructure_appears_in_source(subs, source, dataset_nickname=dset)
        if (n1, n2) != (0,0):
            subs_dict[subs].append((n1, n2))
            subs_dict[subs].append(weight_diff)
    return subs_dict


def substructure_appears_in_source(substr, source, dataset_nickname='bcrp'):
    """
    Returns a tuple (int, int) counting how many compounds of the given source contain the given substructure.
    In position 0 of the tuple, we report the number of inhibitors and in position 1 we report the number of inactives.
    """
    rdkimold = MANYSOURCES_MOLECULES[dataset_nickname]()
    in_class_1 = 0
    in_class_0 = 0
    molid2mol = {}
    molids = rdkimold.molids()
    molid2src = {}
    molid2act = {}
    for molid in molids:
        molid2mol[molid] = rdkimold.molid2mol(molid)
        molid2src[molid] = rdkimold.molid2source(molid)
        molid2act[molid] = rdkimold.molid2label(molid)
    src2molids = defaultdict(list)
    for molid, src in molid2src.iteritems():
        src2molids[src].append(molid)
    # How many molecules contain this substructure in class 1, how many in class 0
    for molid in src2molids[source]:
        molec = molid2mol[molid]
        patt = Chem.MolFromSmarts(substr)
        act = molid2act[molid]
        if molec.HasSubstructMatch(patt):
            if act == 'INHIBITOR':  # Careful, maybe this does not work for all datasets!!
                in_class_1 += 1
            else:
                in_class_0 += 1
    return in_class_1, in_class_0


def faster_relevant_feats(source, dset='bcrp'):
    """
    Here we try to do the same but using Santi's advice
    """
    X, _ = ManysourcesDataset(dset).ecfpsXY(no_dupes=True)   # the sparse matrix of the features
    all_molids = list(ManysourcesDataset(dset).ecfps(no_dupes=True).molids)
    molids_in_source = ManysourcesDataset(dset).molecules().source2molids(source)
    mol_indices = np.array([all_molids.index(molid) for molid in molids_in_source])
    Xsource = X[mol_indices, :]
    features_indices = set(Xsource.indices)
    return features_indices


def plot_smarts(smarts, directory):
    from integration import smartsviewer_utils
    if len(smarts) > 1: # let's remove the C and c...
        svr = smartsviewer_utils.SmartsViewerRunner(w=200, h=200)
        svr.depict(smarts, op.join(directory, smarts + '.png'))
    return op.join(directory, smarts + '.png')


def positive_negative_substructs(model='logreg3', feats='ecfps1', dset='bcrp', lso=True, num_expids=4096,
                                 top_interesting=20):
    '''
    Given a dataset, collect all weights for all substructures across all expids, then average them and check the
    extremes: positive weights mean a substructure that is likely to occur in inhibitors, negative weights mean
    substructures more likely to occur in non-inhibitors. Are we learning something?
    '''
    hub = Hub(dset_id=dset, expids=num_expids, lso=lso, model=model, feats=feats)
    weights, _, expids, foldnums = hub.logreg_models()
    average_weights = np.asarray(weights.mean(axis=0))[0]
    i2s = ManysourcesDataset(dset).ecfps(no_dupes=True).i2s
    order = np.argsort(average_weights)
    ordered_substructures = i2s[order]
    ordered_importances = average_weights[order]
    top_inactives = zip(ordered_importances[0:top_interesting], ordered_substructures[0:top_interesting])
    top_inhibitors = zip(ordered_importances[-top_interesting:], ordered_substructures[-top_interesting:])
    # Let's plot them!
    from PIL import Image
    for weight, substr in top_inactives:
        plot_smarts(substr, '/home/flo/Desktop')
    ims = [Image.open(f) for f in glob.glob(op.join('/home/flo/Desktop', '*.png'))]
    num_lines = math.ceil(float(len(ims))/4)
    blank_image = Image.new("RGB", (800, int(num_lines*200)), color='white')
    for i, im in enumerate(ims):
        im.thumbnail((200,200), Image.ANTIALIAS)
        blank_image.paste(im, (200 * (i%4), 200 * (i/4)))
    blank_image.save(op.join(MANYSOURCES_ROOT, 'data', 'results', dset,
                             'substructs_max_negative_weights_lso.png'))
    for f in glob.glob(op.join('/home/flo/Desktop', '*.png')):
        os.remove(f)
    for weight, substr in top_inhibitors:
        plot_smarts(substr, '/home/flo/Desktop')
    ims = [Image.open(f) for f in glob.glob(op.join('/home/flo/Desktop', '*.png'))]
    num_lines = math.ceil(float(len(ims))/4)
    blank_image = Image.new("RGB", (800, int(num_lines*200)), color='white')
    for i, im in enumerate(ims):
        im.thumbnail((200,200), Image.ANTIALIAS)
        blank_image.paste(im, (200 * (i%4), 200 * (i/4)))
    blank_image.save(op.join(MANYSOURCES_ROOT, 'data', 'results', dset,
                             'substructs_max_positive_weights_lso.png'))
    for f in glob.glob(op.join('/home/flo/Desktop', '*.png')):
        os.remove(f)
    return top_inactives, top_inhibitors


print positive_negative_substructs()
exit(33)


"""
What can we do with this information?
1. compare rankings: are the most weighted substructures (in absolute value) still highly weighted when x source is
in test?
2. check the change in weight for each source across all the expids and do a t-test to check which ones are
significant
3. which substructures are present in the source? Do they show significant change in weight?
4. were those substructures actually important (high absolute value of weight) when the source was part of the
training set?
9. Can we correlate the change of weight (second whiskerplot with only substructures occurring in source) with a worse
prediction of the source? (in terms of average loss for all mols for all splits where the source is in external set)

Comments from Santi:
5. How hard would it be to adapt your code to find out a ranking of features according to how much did they "changed
weight" between "source in" and "source out". That maybe would allow us to highlight concrete features.

6. How hard would it be to restrict this same plot to only subsets of relevant features (substructures) for each source.
And by relevant I mean "substructures that actually appear in the source". For large sources, I would expect not a big
change (because most substructures will be relevant for the source). But for not so big sources, I would expect this
to change the plot dramatically. I think so because I believe that important changes in substructures get hidden by the
overwhelming majority of features that do not care about a source being or not available for training.

7. Can you imagine a way of labelling folds according to whether a molecule was better or worse predicted than average?

8. Can you imagine that performing regression on a weight of a feature we find interesting, using as predictors the
coocurrences, would somehow allow us to explain what coocurrences make a feature important/unimportant?

"""

def test_ranking(weights_in, weights_out):
    import scipy.stats as stats
    return stats.spearmanr(weights_in, weights_out)  # returns r and the p-value associated


def overall_ranking(source, model='logreg3', feats='ecfps1', dset='bcrp'):
    """
    Creates a dictionary and pickles it. Does not return anything. For each expid, computes the Spearman coeff correl
    between the weights in and the weights out (across all substructures)
    """
    if op.exists(op.join(MANYSOURCES_ROOT, 'data', 'results', dset,
                         'spearmans_lso_' + source + '_' + model + '_' + feats + '.dict')):
        return
    spearmans = {}
    expids, all_weights_in, all_weights_out = substructs_weights_one_source(source=source, model=model, feats=feats,
                                                                            dset=dset)
    print len(expids), len(all_weights_in), len(all_weights_out)
    for expid, weights_in, weights_out in zip(expids, all_weights_in, all_weights_out):
        spearmanr, pvalue = test_ranking(weights_in, weights_out)
        spearmans[expid] = (spearmanr, pvalue)
        print expid, spearmanr, pvalue
    with open(op.join(MANYSOURCES_ROOT, 'data', 'results', dset, 'spearmans_lso_' + source + '_' + model + '_' + feats + '.dict'), 'wb') as writer:
        pickle.dump(spearmans, writer, protocol=pickle.HIGHEST_PROTOCOL)


def ranking_relevant_features(source, model='logreg3', feats='ecfps1', dset='bcrp'):
    """
    Same as before but by "relevant features", we mean features that actually occur in the given source
    """
    if op.exists(op.join(MANYSOURCES_ROOT, 'data', 'results', dset, 'spearmans_lso_relfeats_' + source + '_' + model + '_' + feats + '.dict')):
        return
    spearmans = {}
    expids, all_weights_in, all_weights_out = substructs_weights_one_source(source=source, model=model, feats=feats,
                                                                            dset=dset)
    relevant_feature_indexes = list(faster_relevant_feats(source, dset=dset))
    # Select only weights that correspond to relevant features for the given source
    for expid, weights_in, weights_out in zip(expids, all_weights_in, all_weights_out):
        # only to the Spearman test on the relevant feature weights
        spearmanr, pvalue = test_ranking(weights_in[relevant_feature_indexes], weights_out[relevant_feature_indexes])
        spearmans[expid] = (spearmanr, pvalue)
        print expid, spearmanr, pvalue
    with open(op.join(MANYSOURCES_ROOT, 'data', 'results', dset,
                      'spearmans_lso_relfeats_' + source + '_' + model + '_' + feats + '.dict'), 'wb') as writer:
        pickle.dump(spearmans, writer, protocol=pickle.HIGHEST_PROTOCOL)


def plot_spearman_coefs_all_sources(dir, model='logreg3', feats='ecfps1', dset='bcrp'):
    big_dict = {}
    # list all spearman files
    for f in glob.glob(op.join(dir, 'spearmans_lso_*')):
        if not 'relfeats' in op.basename(f):
            source = op.basename(f).partition('_lso_')[2].partition('_logreg')[0]
            print source
            with open(f, 'rb') as reader:
                dict_spearman = pickle.load(reader)
                spearmans = map(lambda x: x[0], dict_spearman.values())
                big_dict[source] = spearmans
    df = pd.DataFrame.from_dict(big_dict)
    tidy_df = pd.melt(df, var_name='source', value_name='Spearman correlation coefficient')
    import seaborn
    seaborn.set_style("ticks")
    seaborn.set_context("talk")
    seaborn.boxplot('source', 'Spearman correlation coefficient', data=tidy_df)
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=90)
    plt.xlabel('Source')
    plt.ylabel('Spearman correlation of feature weights')
    plt.ylim([0,1])
    plt.show()


def plot_spearman_only_relevant_feats_all_sources(dir, model='logreg3', feats='ecfps1', dset='bcrp'):
    big_dict = {}
    # list all spearman files
    for f in glob.glob(op.join(dir, 'spearmans_*')):
        if 'relfeats' in op.basename(f):
            source = op.basename(f).partition('_lso_relfeats_')[2].partition('_logreg')[0]
            print source
            with open(f, 'rb') as reader:
                dict_spearman = pickle.load(reader)
                spearmans = map(lambda x: x[0], dict_spearman.values())
                big_dict[source] = spearmans
    df = pd.DataFrame.from_dict(big_dict)
    tidy_df = pd.melt(df, var_name='source', value_name='Spearman correlation coefficient')
    import seaborn
    seaborn.set_style("ticks")
    seaborn.set_context("talk")
    seaborn.boxplot('source', 'Spearman correlation coefficient', data=tidy_df)
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=90)
    plt.title('Spearman correlations across 4096 experiments, '
              '\nchecking the weights of the relevant features\nwhen the source is in training or in test')
    plt.xlabel('Source')
    plt.ylabel('Spearman correlation of feature weights')
    plt.ylim([0,1])
    plt.show()


def paired_ttest(weights_in, weights_out):
    # TODO: also do it with the bayesian approach
    # Null hypothesis: there is no weight difference.
    from scipy.stats import ttest_1samp
    differences = weights_in - weights_out
    return ttest_1samp(differences, 0) # returns t and the p-value associated


def overall_ttest(source, model='logreg3', feats='ecfps1', dset='bcrp'):
    ttests = {}
    expids, all_weights_in, all_weights_out = substructs_weights_one_source(source=source, model=model, feats=feats,
                                                                            dset=dset)
    for expid, weights_in, weights_out in zip(expids, all_weights_in, all_weights_out):
        t, pvalue = paired_ttest(weights_in, weights_out)
        ttests[expid] = (t, pvalue)
        print expid, t, pvalue
    with open(op.join(MANYSOURCES_ROOT, 'data', 'results', dset, 'paired_ttest_lso_' + source + '_' + model + '_' + feats + '.dict'), 'wb') as writer:
        pickle.dump(ttests, writer, protocol=pickle.HIGHEST_PROTOCOL)


def ttest_per_substructure(source, model='logreg3', feats='ecfps1', dset='bcrp'):
    ttests = {}
    expids, all_weights_in, all_weights_out = substructs_weights_one_source(source=source, model=model, feats=feats,
                                                                            dset=dset)
    print np.array(all_weights_in).shape, np.array(all_weights_out).shape
    i2s = ManysourcesDataset(dset).ecfps(no_dupes=True).i2s
    all_weights_in = list(np.array(all_weights_in).T)
    all_weights_out = list(np.array(all_weights_out).T)
    print len(all_weights_in), len(all_weights_out)
    for i, weights_in in enumerate(all_weights_in):
        ttests[i2s[i]] = paired_ttest(weights_in, all_weights_out[i])
        if i%10 == 0:
            print i2s[i], weights_in
    with open(op.join(MANYSOURCES_ROOT, 'data', 'results', dset, 'paired_ttest_lso_bysubstruct_' + source + '_' + model + '_' + feats + '.dict'), 'wb') as writer:
        pickle.dump(ttests, writer, protocol=pickle.HIGHEST_PROTOCOL)


def do_job_question_3(source, model='logreg3', feats='ecfps1', dset='bcrp', significant=0.01):
    # I really should make it faster
    # Read the t-test per substructure file
    significant_substructures = []
    with open(op.join(MANYSOURCES_ROOT, 'data', 'results', dset, 'paired_ttest_lso_bysubstruct_' + source + '_' + model + '_' + feats + '.dict'), 'rb') as reader:
        ttests = pickle.load(reader)
        for substructure, ttest_res in ttests.iteritems():
            if ttest_res[1] <= significant:
                if substructure_appears_in_source(substructure, source, dataset_nickname=dset) != (0,0):
                    print substructure, ttest_res[1]
                    significant_substructures.append(substructure)
    return significant_substructures


def analyse_most_changing_substructs(source, dset='bcrp', model='logreg3', feats='ecfps1', top=10,
                                     temp_dir='/home/floriane/Desktop'):
    """
    Check which substructures are in the source among all those that were showing important changes in weight. Plots
    the substructure using SmartsViewer.
    """
    substructs_dict = most_changing_substructs_source(dset, model=model, feats=feats, source=source, top=top)
    distinct_subs = len(substructs_dict)
    indices_in_source = list(faster_relevant_feats(source, dset))
    i2s = ManysourcesDataset(dset).ecfps(no_dupes=True).i2s
    substructs_in_source = i2s[indices_in_source]
    in_source_and_changing = [sub for sub in substructs_dict.keys() if sub in substructs_in_source ]
    print in_source_and_changing
    print "Proportion of substructures that most change in weight that actually appear in %s: %.2f" % \
          (source, float(len(in_source_and_changing))/distinct_subs)
    from chemdeco.integration import smartsviewer_utils
    from PIL import Image
    for substr in in_source_and_changing:
        if len(substr) > 1: # let's remove the C and c...
            svr = smartsviewer_utils.SmartsViewerRunner(w=200, h=200)
            svr.depict(substr, op.join(temp_dir, substr + '.png'))

    ims = [Image.open(f) for f in glob.glob(op.join(temp_dir, '*.png'))]
    num_lines = math.ceil(float(len(ims))/4)
    blank_image = Image.new("RGB", (800, int(num_lines*200)), color='white')
    for i, im in enumerate(ims):
        im.thumbnail((200,200), Image.ANTIALIAS)
        blank_image.paste(im, (200 * (i%4), 200 * (i/4)))
    blank_image.save(op.join(MANYSOURCES_ROOT, 'data', 'results', dset, 'substructs_in_%s_max_change_weight_%s_%s.png'
                             %(source, model, feats)))
    # TODO: automatically remove the images from the temp_dir


def barplot_most_changing_substructs(dset='bcrp', model='logreg3', feats='ecfps1', top=10):
    """
    Plots a 2-bar per source bar plot: first bar corresponds to the amount of substructures that most changed weight in
    the in / out experiments. Second bar corresponds to the amount of those substructures that actually occur in the
    source
    """
    values_total = []
    values_insource = []
    sources = ManysourcesDataset(dset='bcrp').molecules().present_sources()
    values_dict = {}
    for source in sources:
        print source
        substructs_dict = most_changing_substructs_source(dset, model=model, feats=feats, source=source, top=top)
        values_total.append(len(substructs_dict))
        indices_in_source = list(faster_relevant_feats(source, dset))
        i2s = ManysourcesDataset(dset).ecfps(no_dupes=True).i2s
        substructs_in_source = i2s[indices_in_source]
        in_source_and_changing = [sub for sub in substructs_dict.keys() if sub in substructs_in_source]
        values_insource.append(len(in_source_and_changing))
    values_dict['source'] = list(sources)
    values_dict['total'] = values_total
    values_dict['in source'] = values_insource

    ind = np.arange(len(sources))  # the x locations for the groups
    width = 0.35       # the width of the bars
    import seaborn
    seaborn.set_style("ticks")
    seaborn.set_context("talk")
    seaborn.set_palette('deep')
    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, values_dict['total'], width, color='0.75')
    rects2 = ax.bar(ind+width, values_dict['in source'], width)
    ax.legend((rects1[0], rects2[0]), ('Total', 'In source only') )
    locs, labels = plt.xticks(map(lambda x: x + width, ind), values_dict['source'])
    plt.setp(labels, rotation=90)
    plt.xlabel('Source')
    plt.ylabel('Number of substructures most changing weight')
    plt.ylim([0,320])
    plt.show()


def losses_correl_weight_changes(dset='bcrp', model='logreg3', feats='ecfps1', expids=tuple(range(4096)), calib="'0.1'"):
    """
    Plots the correlation between the average loss per source with the average relevant spearman correlation per source.
    """

    # Copied from Santi but then changed a bit to fit my needs. Reads the losses for the given model
    def read_cached():
        cache_path = op.join(MANYSOURCES_ROOT, 'data', 'results', 'square_losses.h5')
        result_coords = '/dset=bcrp/feats=ecfps1/model=logreg3/lso=True/score_calibration=\'0-1\''
        with h5py.File(cache_path, 'r') as h5:
            group = h5[result_coords]
            infile_expids = group['expids'][()] if expids is not None else expids
            if 0 == len(set(expids) - set(infile_expids[:, 0])):
                e2r = {e: i for e, i in infile_expids if i >= 0}
                ok_expids = [expid for expid in expids if expid in e2r]
                rows = [e2r[expid] for expid in ok_expids]
                losses = group['losses'][rows].T
                molids = group['molids'][:]
                return pd.DataFrame(losses, columns=ok_expids, index=molids)

    losses = read_cached()
    molids = list(losses.index)
    #print molids
    equivalence_to_source = ManysourcesDataset(dset).mols().molids2sources(molids)
    losses['source'] = equivalence_to_source
    df_mean_loss = losses.groupby('source').mean()  # average loss per source per expid
    dict_mean_loss = defaultdict(float)
    for src in df_mean_loss.index:
        dict_mean_loss[src] = np.array(list(df_mean_loss.loc[src])).mean()
    df_mean_loss = pd.DataFrame.from_dict(dict_mean_loss, orient='index')
    df_mean_loss.columns = ['average_loss']

    big_dict = {}
    # list all spearman files
    for f in glob.glob(op.join(op.join(MANYSOURCES_ROOT, 'data', 'results', 'bcrp'), 'spearmans_*')):
        if 'relfeats' in op.basename(f):
            source = op.basename(f).partition('_lso_relfeats_')[2].partition('_logreg')[0]
            #print source
            with open(f, 'rb') as reader:
                dict_spearman = pickle.load(reader)
                spearmans = map(lambda x: x[0], dict_spearman.values())
                big_dict[source] = np.array(spearmans).mean()

    df_mean_loss['spearmans'] = [big_dict[source] for source in df_mean_loss.index]
    print df_mean_loss

    # plot correlation
    import seaborn
    seaborn.set_style("ticks")
    seaborn.set_context("talk")
    seaborn.set_palette('deep')
    seaborn.set_context(rc={'lines.markeredgewidth': 0.1})  # otherwise we only see regression line, see
    #http://stackoverflow.com/questions/26618339/new-version-of-matplotlib-with-seaborn-line-markers-not-functioning
    seaborn.lmplot('spearmans', 'average_loss', df_mean_loss,  scatter_kws={"marker": ".", 's': 50})
    #plt.scatter([big_dict[src] for src in sources], [dict_mean_loss[src] for src in sources])
    plt.xlabel('Spearman coefficient correlation of feature importances')
    plt.ylabel('Average loss when source is in external set')
    plt.title('Absence of correlation between hardness to predict (high loss) \nand high change in feature weights '
              'at the source level')
    plt.show()


def relationship_spearman_size_source(dir, model='logreg3', feats='ecfps1', dset='bcrp'):
    """
    Plots the relationship between the size of the source vs the average relevant Spearman corr coeff. One point per
    source on the plot.
    """
    small_dict = defaultdict(list)
    # list all spearman files
    for f in glob.glob(op.join(dir, 'spearmans_*')):
        if 'relfeats' in op.basename(f):
            source = op.basename(f).partition('_lso_relfeats_')[2].partition('_logreg')[0]
            print source
            small_dict['source'].append(source)
            small_dict['size'].append(len(ManysourcesDataset(dset).mols().sources2molids([source])))
            with open(f, 'rb') as reader:
                dict_spearman = pickle.load(reader)
                spearmans = map(lambda x: x[0], dict_spearman.values())
                small_dict['average spearman'].append(np.mean(np.array(spearmans)))
    df = pd.DataFrame.from_dict(small_dict)
    import seaborn
    seaborn.set_style("ticks")
    seaborn.set_context("talk")
    seaborn.lmplot('size', 'average spearman', data=df, scatter_kws={"marker": "o", "color": "slategray"},
                   line_kws={"linewidth": 1, "color": "seagreen"})
    plt.show()



if __name__ == '__main__':
    #sources = ManysourcesDataset(dset='hERG').molecules().present_sources()
    #for source in sources:
    #    print source
    #    most_changing_substructs_source(source=source, dset='bcrp', top=10)
        #ttest_per_substructure(source)
    #plot_spearman_coefs_all_sources(op.join(MANYSOURCES_ROOT, 'data', 'results', 'bcrp'))
    #do_job_question_3('Zembruski_2011')
    #cache_relevant_features_all_sources()
    #plot_spearman_only_relevant_feats_all_sources(op.join(MANYSOURCES_ROOT, 'data', 'results', 'bcrp'))
    #plot_spearman_coefs_all_sources(op.join(MANYSOURCES_ROOT, 'data', 'results', 'bcrp'))
    #barplot_most_changing_substructs()
    #proportion_relevant_features('flavonoids_Zhang_2004')
    #print substructures_change_weight('flavonoids_Zhang_2004')
    #relationship_spearman_size_source(op.join(MANYSOURCES_ROOT, 'data', 'results', 'bcrp'))
    source = 'Ochoa-Puentes_2011'
    analyse_most_changing_substructs(source, dset='bcrp')
