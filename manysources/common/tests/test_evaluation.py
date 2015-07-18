# coding=utf-8
from itertools import chain

import pytest
import numpy as np

from manysources.common.evaluation import cv_splits, stratified_random_subset, random_subset, JaggedCV, GroupsSplitCV


def test_cv_splits_no_stratify():

    total = 10
    n_folds = 5
    cver = cv_splits(total, n_folds, stratify=False)

    # Each fold contains the whole set without repetitions
    for fold in xrange(n_folds):
        train, trest = cver(fold)
        in_train = set(train)
        in_trest = set(trest)
        assert len(in_train & in_trest) == 0, 'There must be no repetition in train/test'
        assert len(in_train | in_trest) == total, 'All the examples are in the split'

    # Each example is exactly once in test
    in_trest = list(chain(*[cver(fold)[1] for fold in xrange(n_folds)]))
    assert len(set(in_trest)) == len(in_trest), 'Each example must be tested exactly once'
    assert len(in_trest) == total, 'Each example must be in test at least once'

    # We get an exception if we try to get stratified sampling without providing Y
    with pytest.raises(Exception):
        cv_splits(10, n_folds)

    # We get an exception if we try to get an unexistent fold
    with pytest.raises(Exception):
        cv_splits(10, n_folds)(11)

    # We get an exception if we try to get more folds than instances
    with pytest.raises(Exception):
        cv_splits(5, 6, stratify=False)


def test_cv_splits_stratified():

    total = 10
    n_folds = 5

    # Stratify test
    Y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    cver = cv_splits(10, n_folds, Y=Y)
    balancy_train = []
    for fold in range(n_folds):
        balancy_train.append(np.sum(Y[cver(fold)[0]])/float(len(cver(fold)[0])))
    # the average balancy should be around 0.5
    assert 0.48 < np.mean(np.array(balancy_train))
    assert 0.52 > np.mean(np.array(balancy_train))

    # Banned train
    cver = cv_splits(10, n_folds, stratify=False, banned_train=np.array([1]))
    for fold in range(n_folds):
        assert 1 not in cver(fold)[0]

    # Banned train 2
    cver = cv_splits(20, n_folds, stratify=False, banned_train=[0, 1])
    for fold in range(n_folds):
        assert 1 not in cver(fold)[0]
        assert 0 not in cver(fold)[0]

    # Banned test
    cver = cv_splits(10, n_folds, stratify=False, banned_test=np.array([1]))
    for fold in range(n_folds):
        assert 1 not in cver(fold)[1]

    # Banned test 2
    cver = cv_splits(10, n_folds, stratify=False, banned_test=[0, 1])
    for fold in range(n_folds):
        assert 1 not in cver(fold)[1]
        assert 0 not in cver(fold)[1]


def test_stratified_random_subset():
    # 1/3 of the instances belong to group 0
    groups = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    selected = stratified_random_subset(groups, np.random.RandomState(0))
    # check size of the selection
    assert len(selected)/float(len(groups)) == 0.4
    # check stratification
    # TODO: like for random_subset, do it many times!
    assert np.sum(groups[list(selected)])/float(len(selected)) > 0.6
    assert np.sum(groups[list(selected)])/float(len(selected)) < 0.7


def test_random_subset():
    # Just getting Santi's doctest
    rng = np.random.RandomState(0)
    num_samples = 100000
    subsets_sizes = [len(random_subset(10, rng, pick_probability=0.4)) for _ in xrange(num_samples)]
    assert np.abs(np.mean(subsets_sizes) - 4) < 0.01
    subsets_sizes = [len(random_subset(10, rng, pick_probability=0.05)) for _ in xrange(num_samples)]
    assert np.abs(np.mean(subsets_sizes) - 0.5) < 0.01


def test_jagged_cv():
    fold_sizes = [10, 10, 5, 30]
    whole_size = 10 + 10 + 5 + 30
    jcv = JaggedCV(seed=0, fold_sizes=fold_sizes)
    in_test = []
    for i, splitter in enumerate(jcv.splitters()):
        assert len(splitter.split()[1]) == fold_sizes[i]  # we get the sizes we asked for
        in_test += list(splitter.split()[1])
    assert len(in_test) == whole_size
    for i in range(whole_size):
        assert i in in_test


def test_groupsplit_cv():
    # 4 groups, 20 examples
    groups =[0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]
    seed = 0
    num_folds = 4
    gscv = GroupsSplitCV(groups, seed=seed, num_folds=num_folds)

    # Each example is exactly once in test
    in_test = list(chain(*[splitter.split()[1] for splitter in gscv.splitters()]))
    assert len(set(in_test)) == len(in_test), 'Each example must be tested exactly once'
    assert len(in_test) == 20, 'Each example must be in test at least once'

    # Each fold contains the whole set without repetitions
    for splitter in gscv.splitters():
        train, trest = splitter.split()
        in_train = set(train)
        in_trest = set(trest)
        assert len(in_train & in_trest) == 0, 'There must be no repetition in train/test'
        assert len(in_train | in_trest) == 20, 'All the examples are in the split'

    # No groups are split accross train and test
    for splitter in gscv.splitters():
        train, trest = splitter.split()
        groups_in_train = set([groups[i] for i in train])
        groups_in_test = set([groups[i] for i in trest])
        assert len(groups_in_test & groups_in_train) == 0, 'There must be no repetition of groups in train/test'
        assert len(groups_in_test | groups_in_train) == 4, 'All groups must be present in the split'

    # We get an exception if we try to get more folds than groups
    with pytest.raises(Exception):
        GroupsSplitCV(groups, num_folds=5)



#if __name__ == '__main__':
#    test_groupsplit_cv()