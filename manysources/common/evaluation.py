# coding=utf-8
from numbers import Integral

import numpy as np


# --- Cross-validation and random splitting
from whatami import whatable


def cv_splits(num_points,
              num_folds,
              rng=None,
              stratify=True,
              Y=None,
              banned_train=None,
              banned_test=None):
    """
    (Stratified) cross-validation splits
    (copied from oscail.common.evaluation).

    Parameters
    ----------
    num_points: int
        the number of elements to split

    num_folds: int
        the number of splits

    rng: python/numpy random number generator, default None
        it will be used to shuffle the data; if None,

    Y: list or numpy array,
        the group for each point (e.g. the class, the score, the source...)

    stratify: boolean, default True
        if True and Y is given, a best effort is carried to keep consistent the Y proportions
        in each split when dealing with classification

    banned_train: list or numpy array of integers, default None
        indices to not include in train (e.g. non-target class for OCC)

    banned_test: list or numpy array of integers, default None
        indices to not include in test (e.g. extra data used to build but not to test)

    Returns
    -------
    A function that maps a fold index (from 0 to num_folds-1) to the indices of its train/test instances.
    """
    if num_folds > num_points:
        raise Exception('Cannot provide more folds than given number of instances')
    if rng is None:
        rng = np.random.RandomState(0)
    permutation = rng.permutation(num_points)
    if stratify:
        if Y is not None:
            permutation = permutation[np.argsort(Y[permutation])]  # Indices randomized but sorted by group
        else:
            raise Exception('Stratify requires the data groups, but they were not provided')
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
        # N.B. the extra shuffling might hide info leaking,
        #      but more often than not, it avoid artifact results due to ordering in the original dataset + stablesort
        test_indices = folds[fold][rng.permutation(len(folds[fold]))]
        ban_in_train = set(test_indices) | banned_train
        train_indices = [train_i for train_i in rng.permutation(num_points)
                         if train_i not in ban_in_train]
        test_indices = [test_i for test_i in test_indices if test_i not in banned_test]
        return train_indices, test_indices

    return cver


def random_subset(num_elements, rng, pick_probability=0.4):
    """Returns a random subset (without replacement) from a superset of naturals, from 0 to num_elements-1.
    Examples:
    >>> rng = np.random.RandomState(0)
    >>> num_samples = 100000
    >>> subsets_sizes = [len(random_subset(10, rng, pick_probability=0.4)) for _ in xrange(num_samples)]
    >>> np.abs(np.mean(subsets_sizes) - 4) < 0.01
    True
    >>> subsets_sizes = [len(random_subset(10, rng, pick_probability=0.05)) for _ in xrange(num_samples)]
    >>> np.abs(np.mean(subsets_sizes) - 0.5) < 0.01
    True
    """
    return set(np.where(rng.uniform(size=num_elements) < pick_probability)[0])


def stratified_random_subset(groups,
                             rng,
                             pick_probability=0.4,
                             min_per_group=0):
    # (forced) Stratified + minimum selection size
    indices = np.arange(len(groups))
    selected = set()
    for group in np.unique(groups):
        group_indices = indices[groups == group]
        to_pick = len(group_indices) * pick_probability  # e.g. 36.4
        # Now, to transform to a size:
        #   36.4 -> 36 (40% of the times) or 37 (60% of the times)
        # Having only two options might not be the best option,
        # but it is not terrible either
        # (we are anyway already biasing a lot the nature of subsamples)
        to_pick = np.ceil(to_pick) if rng.uniform() < np.modf(to_pick)[0] \
            else np.floor(to_pick)
        # We also have minimum pick requirements
        to_pick = min(max(min_per_group, to_pick), len(group_indices))
        # Choose
        selected.update(rng.choice(group_indices, to_pick, replace=False))
    return selected


#############################
# "Coordinated" splitters
#############################

# --- Standard (Stratified) Cross-Validation

@whatable
class CrossValidationEval(object):

    __slots__ = ('seed', 'num_folds')

    def __init__(self,
                 num_points,
                 seed=0,
                 num_folds=10,
                 Y=None,
                 banned_train=None,
                 banned_test=None):
        self.seed = seed
        self.num_folds = num_folds
        self.stratify = Y is not None  # ...CHECK: and it is a classification problem...
        self.banned_train = banned_train
        self.banned_test = banned_test
        self._partitioner = cv_splits(num_points,
                                      self.num_folds,
                                      np.random.RandomState(self.seed),
                                      stratify=self.stratify,
                                      Y=Y,
                                      banned_train=self.banned_train,
                                      banned_test=self.banned_test)

    def splitters(self):
        return (CrossValidationFold(self, fold_num) for fold_num in xrange(self.num_folds))

    def __iter__(self):
        return self.splitters()


@whatable
class CrossValidationFold(object):

    __slots__ = ('cv', 'fold_num')

    def __init__(self, cv, fold_num):
        self.cv = cv
        self.fold_num = fold_num

    def splitters(self):
        return self,

    def split(self):
        return self.cv.partitioner(self.fold_num)


# --- Classical Random Splitting

@whatable
class RandomSplitEval(object):

    def __init__(self, total_size, test_size, seed):
        super(RandomSplitEval, self).__init__()
        self.total_size = total_size
        self.test_size = test_size
        self.seed = seed

    def splitters(self):
        return self,

    def split(self):
        rng = np.random.RandomState(self.seed)  # N.B. this will allow only whatever int max different partitions
        indices = rng.permutation(self.total_size)
        return indices[self.test_size:], indices[:self.test_size]


# --- Classical Random Splitting (ala CV)

@whatable
class JaggedCV(object):

    def __init__(self, seed, fold_sizes):
        super(JaggedCV, self).__init__()
        self.seed = seed
        self.fold_sizes = fold_sizes
        self._permutation = np.random.RandomState(self.seed).permutation(np.sum(fold_sizes))
        self._split_points = np.hstack(([0], np.cumsum(fold_sizes)))

    def splitters(self):
        return (JaggedCVFold(self, fold_num) for fold_num in xrange(len(self.fold_sizes)))


@whatable
class JaggedCVFold(object):

    __slots__ = ('cv', 'fold_num')

    def __init__(self, cv, fold_num):
        self.cv = cv
        self.fold_num = fold_num

    def split(self):
        sp1 = self.cv._split_points[self.fold_num]
        sp2 = self.cv._split_points[self.fold_num + 1]
        return np.hstack((self.cv._permutation[:sp1], self.cv._permutation[sp2:])), \
            self.cv._permutation[sp1:sp2]


# --- Grouped Cross-Validation (aka cross-validation over groups)

@whatable
class GroupsSplitCV(object):
    """
    Splits a dataset in which each example pertains to one group.
    """

    def __init__(self, example2group, seed=0, num_folds=5):
        super(GroupsSplitCV, self).__init__()
        self.seed = seed
        self.num_folds = num_folds
        self._ex2group = example2group
        self._present_groups = sorted(set(example2group))
        self._rng = np.random.RandomState(self.seed)
        self._partitioner = cv_splits(
            num_points=len(self._present_groups),
            num_folds=self.num_folds,
            rng=self._rng,
            stratify=False
        )
        if num_folds > len(self._present_groups):
            raise Exception('Cannot provide more folds than existing groups')

    def splitters(self):
        return (GroupsSplitCVFold(self, fold_num) for fold_num in xrange(self.num_folds))

    def __iter__(self):
        return self.splitters()


@whatable
class GroupsSplitCVFold(object):

    __slots__ = ('cv', 'fold_num')

    def __init__(self, cv, fold_num):
        super(GroupsSplitCVFold, self).__init__()
        self.cv = cv
        self.fold_num = fold_num

    def split(self):
        subset = [self.cv._present_groups[i] for i in self.cv._partitioner(self.fold_num)[1]]
        dev_set = [i for i in xrange(len(self.cv._ex2group)) if self.cv._ex2group[i] not in subset]
        ext_set = [i for i in xrange(len(self.cv._ex2group)) if self.cv._ex2group[i] in subset]
        return np.array(dev_set), np.array(ext_set)


# --- Grouped Random Splitting

@whatable
class GroupsSplitEval(object):
    """Splits a dataset so that all the examples in a group go to the same split.

    In other words: each example is in a group and each group goes to just one of train or test.

    Optionally we allow stratified splitting of the groups
    (when groups are in turn assigned a meta-group).

    Parameters
    ----------
    example2group: list or numpy array
        The ith element of the list contains the group assignment for element i

    seed: int, defaults to 0
        The seed used to initialize the random number generator used to shuffle and select.
        Note that the same partition will be generated regardless how many times partition() is called
        (i.e. we restart the rng at each call to partition())
        but we still can use accept_function to force different partitions to be generated
        (not very efficiently though)

    ext_proportion: float in [0,1], default 0.1
        The proportion of groups to keep

    meta_groups: list or numpy array
        A meta-group assignment for each group (size=len(set(groups)));
        if provided, a best-effort is made to keep meta-group proportions in training and testing.

    accept_function: function (numpy array of ints) -> bool, defaults to f(group) -> True
        This function is invoked upon generation of a new partition (called with subset of groups in test);
        only a group that evaluates to True will be returned (warning, this could lead to an infinite loop)
        An example is "GroupsSplitEval.accept_unique()", that ensures that no partition is returned twice.

    Examples
    --------
    >>> groups = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    >>> groups_Y = np.array([0, 0, 0, 1, 1])
    """

    # Many issues, but practical

    def __init__(self,
                 example2group,
                 seed=0,
                 ext_size=0.1,
                 group2metagroup=None,
                 min_per_group=0,
                 accept_function=lambda subset: True):
        super(GroupsSplitEval, self).__init__()
        self.seed = seed
        self.has_meta_groups = group2metagroup is not None
        self.min_per_group = min_per_group if self.has_meta_groups else None  # Should maybe warn of incompatible params
        self._ex2group = example2group
        # We do not assume anything about the groups
        self._present_groups = sorted(set(example2group))
        self._i2g = {i: g for i, g in enumerate(self._present_groups)}
        self._meta_groups = group2metagroup
        self._accept = accept_function
        # Finally, we ad-hocish compute the external size
        self.ext_size = ext_size if isinstance(ext_size, Integral) else \
            int(np.round(len(self._present_groups) * ext_size))

    def splitters(self):
        return self,

    def split(self):
        rng = np.random.RandomState(self.seed)  # N.B. this will allow only as many partitions as MAXINT
        while True:
            # subset = random_subset(len(self._present_groups),
            #                        rng=rng,
            #                        pick_probability=self.ext_size)
            subset = rng.choice(np.arange(len(self._present_groups)),
                                size=self.ext_size,
                                replace=False)
            subset = set(map(self._i2g.get, subset))
            if self._accept(subset):
                dev_set = [i for i in xrange(len(self._ex2group)) if self._ex2group[i] not in subset]
                ext_set = [i for i in xrange(len(self._ex2group)) if self._ex2group[i] in subset]
                return np.array(dev_set), np.array(ext_set)

    @staticmethod
    def accept_unique():

        subsets = set()

        def accept(subset):
            if not tuple(sorted(subset)) in subsets:
                subsets.add(tuple(sorted(subset)))
                return True
            return False

        return accept