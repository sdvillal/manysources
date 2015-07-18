# coding=utf-8
"""Results generation logic for the ManySources paper experiments.

Quick and dirty stuff to help Flo getting started.
Should be slowly superceeded by stuff in hub.

Each result is identified by these coordinates:
  - expid: an integer that is used to seed random number generators and identify LSO partitions
  - dset: a string that identifies the molecules dataset being used (e.g. 'bcrp')
  - feats: a string representing a concrete feature-based representations for the molecules
  - model: a string representing a concrete model configuration
"""
from copy import copy
from functools import partial
from glob import glob
from itertools import product
from time import time, strftime
from traceback import format_exc
import os.path as op
from user import home

import h5py
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.metrics import roc_auc_score
import numpy as np
from argh import arg
from whatami import whatareyou
from whatami.wrappers.what_sklearn import whatamise_sklearn

from manysources.common.misc import ensure_dir
from manysources.datasets import ManysourcesDataset, manysources_results_root, MANYSOURCES_MOLECULES
from manysources.evaluation import setup_splitters_lsocv, crscv_from_lsocv


# Probably, the expids we tried

DEFAULT_EXPIDS = tuple(range(32768))

# --- Make scikit learn models whatable

whatamise_sklearn()


# --- Models specs

def model_config_string(model):
    return whatareyou(model).id()


class LogregHDF5Storer(object):

    @staticmethod
    def to_hdf5(logreg, group, compression='lzf'):
        group.create_dataset('logreg_coef', data=logreg.coef_, compression=compression)
        group['logreg_intercept'] = logreg.intercept_
        # We could save in sparse format depending on the sparsity of the coef vector...

    @staticmethod
    def from_hdf5(group):
        return group['logreg_coef'][:], group['logreg_intercept'][:]
        # N.B. intercept is an array


class RFHDF5Storer(object):

    @staticmethod
    def to_hdf5(rf, group, compression='lzf'):
        group.create_dataset('rf_feature_importances', data=rf.feature_importances_, compression=compression)

    @staticmethod
    def from_hdf5(group):
        return group['rf_feature_importances'][:]
        # N.B. intercept is an array


class ModelConfig(object):
    """The surrender to the failure of the unfinished Configurable, I will take my revenge..."""

    def __init__(self, nickname, name, model, storer):
        super(ModelConfig, self).__init__()
        self.nickname = nickname
        self.name = name
        self.model = model
        self.storer = storer

    def seed_model(self, seed):
        if isinstance(self.model, RandomForestClassifier):  # hack, revisit
            return self.model.__class__(random_state=seed)
        return self.model  # superhack


MANYSOURCES_MODELS = {
    'logreg1': ModelConfig('logreg1',
                           'logreg#penalty=l2#C=1',
                           LogisticRegression(penalty='l2', C=1, random_state=0),
                           LogregHDF5Storer),
    'logreg3': ModelConfig('logreg3',
                           'logreg#penalty=l1#C=1',
                           LogisticRegression(penalty='l1', C=1, random_state=0),
                           LogregHDF5Storer),
    'logreg5': ModelConfig('logreg5',
                           'logreg#penalty=l1#C=2',
                           LogisticRegression(penalty='l1', C=2, random_state=0),
                           LogregHDF5Storer),
    'logreg7': ModelConfig('logreg7',
                           'logreg#penalty=l1#C=0.5',
                           LogisticRegression(penalty='l1', C=0.5, random_state=0),
                           LogregHDF5Storer),
    #
    # Unreasonably bad performance
    #
    # 'logreg2': ModelConfig('logreg2',
    #                        'logreg#penalty=l2#C=1E-4',
    #                        LogisticRegression(penalty='l2', C=1E-4, random_state=0),
    #                        LogregHDF5Storer),
    #
    # 'logreg4': ModelConfig('logreg4',
    #                        'logreg#penalty=l1#C=1E-4',
    #                        LogisticRegression(penalty='l1', C=1E-4, random_state=0),
    #                        LogregHDF5Storer)
    #

    'rfc1':  ModelConfig('rfc1',
                         'rfc#n_estimators=200',
                         RandomForestClassifier(n_estimators=200, random_state=0),
                         RFHDF5Storer),
}


# --- Features specs


def extract_ecfps(dset, no_dupes=True, binary=True):
    molids, X, y = dset.ecfps_molidsXY(no_dupes=no_dupes)
    if binary:
        X = X.copy()
        X.data = np.ones(len(X.data))   # LAME provenance, should be Configurable
    return molids, X, y


def extract_manyhot_sources(molids, dset, manyhot=True):
    def all_sources(rdkmol):
        # We only have this information for BCRP
        other_sources = []
        try:
            if manyhot:
                other_sources = rdkmol.GetProp('DUPLICATES')
                other_sources = other_sources.split('\n')
                other_sources = [dup.split('=')[1].split('__')[0] for dup in other_sources]
        finally:
            return [rdkmol.GetProp('ActivitySource')] + other_sources
    mols = dset.mols()
    sources = mols.i2sources_order()
    Xsrc = np.zeros((len(molids), len(sources)), dtype=np.bool)
    for row, molid in enumerate(molids):
        mol_sources = all_sources(mols.molid2mol(molid))
        Xsrc[row, :] = np.in1d(sources, mol_sources)
    return Xsrc


def extract_ecfps_and_manyhot_sources(dset, no_dupes=True, binary=True, manyhot=True):
    molids, X, y = extract_ecfps(dset, no_dupes=no_dupes, binary=binary)
    Xsrc = extract_manyhot_sources(molids, dset, manyhot=manyhot)
    return molids, hstack((X, csr_matrix(Xsrc))).tocsr(), y


def extract_rdkdescs(dset):
    molids, _, X, y = dset.rdkdescs()
    return molids, X, y  # FIXME: we should also store feature names in HDF5s


def extract_rdkdescs_and_manyhot_sources(dset, manyhot=True):
    molids, X, y = extract_rdkdescs(dset)
    Xsrc = extract_manyhot_sources(molids, dset, manyhot=manyhot)
    return molids, np.hstack((X, Xsrc)), y


class FeaturesConfig(object):
    """The surrender to the failure of the unfinished Configurable, I will take my revenge..."""

    def __init__(self, nickname, name, extractor, preprocessors=()):
        super(FeaturesConfig, self).__init__()
        self.nickname = nickname
        self.name = name
        self.extractor = extractor
        self.preprocessors = preprocessors


# features_short_name -> (extractor, *[preprocesors]*)
MANYSOURCES_FEATS = {
    'ecfps1': FeaturesConfig('ecfps1', 'ecfps_nodupes_binary', extract_ecfps),
    'rdkdescs1': FeaturesConfig('rdkdescs1', 'all_rdkit_descriptors_noprepro', extract_rdkdescs),

    # Add sources as features, 1-hot encoding
    'ecfps1+source': FeaturesConfig('ecfps1+source', 'ecfps_nodupes_binary_allsourcesonehot',
                                    partial(extract_ecfps_and_manyhot_sources, manyhot=False)),
    'ecfps1+sources': FeaturesConfig('ecfps1+source', 'ecfps_nodupes_binary_allsourcesonehot',
                                     partial(extract_ecfps_and_manyhot_sources, manyhot=True)),
    'rdkdescs1+source': FeaturesConfig('rdkdescs1', 'all_rdkit_descriptors_noprepro',
                                       partial(extract_rdkdescs_and_manyhot_sources, manyhot=False)),
    'rdkdescs1+sources': FeaturesConfig('rdkdescs1', 'all_rdkit_descriptors_noprepro',
                                        partial(extract_rdkdescs_and_manyhot_sources, manyhot=True)),
}


# --- Results coordinates in disk

def single_hdf5_per_exp(expid=0, dset='bcrp', feats='ecfps1', model='logreg1'):
    expid_root = op.join(manysources_results_root(dset), 'expid=%d' % expid)
    return op.join(expid_root, 'dset=%s#feats=%s#expid=%d#model=%s.h5' % (dset, feats, expid, model))


def merged_hdf5_file(expid=0, dset='bcrp', feats='ecfps1', model='logreg1'):
    # Keep expid even if unused
    return op.join(manysources_results_root(dset), 'feats=%s#model=%s.h5' % (feats, model))


# --- Result generation logic


@arg('-e', '--expids', default=(0,), nargs='+', type=int, help='The experiment-ids to run')
def generate_lsocv_results(dset='bcrp',
                           model='logreg1',
                           feats='ecfps1',
                           expids=(0,),
                           compression='lzf',
                           reraise=False):
    """Runs train/test experiments for the "manysources" experiments, saving the results in disk.

    Each individual experiment is saved to an HDF5 file. Afterwards they can be merged.

    The generated HDF5 files have the following structure:
      /dsets
        /bcrp{config}
      /models
        /logreg1{config}
      /featss
        /ecfps1{config}
      /dset='bcrp'
        /feats='ecps1'
          /ids=[molid1, molid2, ..., molidn]
          /expid=3
            /lsocv{config, num_folds}
              /fold0'{config}
                /ext_indices
                /ext_scores
                /y_ext
                /DONE or FAILED
                /model='logreg3'{config, train_time, test_time, auc}
                  /model_data1
                  /model_data2
                  /...
              /fold=1...
              /...
            /crscv#seed=3{config}
                /*same as lsocv*

    Parameters
    ----------
    dset: string, default 'bcrp'
        the name of the dataset
    model: string, default 'logreg1'
        the name of the model configuration
    feats: string, default 'ecfp1'
        the name of the molecular features
    expids: iterable of ints, default (1,)
        the experiment ids to carry
    compression: string, default 'lzf'
        the compression used to store arrays in the HDF5 file; if None, no compression will be used
    reraise: boolean, default False
        if True, exceptions are raised; if not, they are ignored and results keep being generated

    Returns
    -------
    Nothing, but prints a cool DONE and saves the results in disk.
    """

    def train_test(model, X, y, train_indices, trest_indices):
        # Split
        Xtrain, ytrain, Xtrest, ytrest = \
            X[train_indices, :], y[train_indices], \
            X[trest_indices, :], y[trest_indices]
        # Train
        model = copy(model)
        start = time()
        model.fit(Xtrain, ytrain)
        train_time = time() - start
        # Test
        start = time()
        scores = model.predict_proba(Xtrest)[:, 1]
        test_time = time() - start
        return scores, model, train_time, test_time

    def generate_split_result(model_config, X, y, split_id, splitter, cvgroup, reraise=False):
        # Splitting
        train_indices, trest_indices = splitter.split()
        fold_group = cvgroup.require_group(split_id)
        try:
            fold_group.attrs['config'] = splitter.what().id()
            fold_group.create_dataset('test_indices', data=trest_indices, compression=compression)  # uncompressible
            fold_group.create_dataset('y_test', data=y[trest_indices], compression=compression)
        except:
            pass  # Dodgy
        # Model configuration
        model_group = fold_group.require_group('model=%s' % model_config.nickname)
        try:
            # already done?
            if 'DONE' in fold_group.keys():
                print '%s already done, skipping...' % model_group.name
                return
            if 'FAILED' in fold_group.keys():
                print '%s already failed, skipping...' % model_group.name
                return
            # compute the result
            scores, model, train_time, test_time = \
                train_test(model_config.seed_model(expid), X, y, train_indices, trest_indices)
            # save scores, auc, times
            try:
                model_group.attrs['auc'] = roc_auc_score(y[trest_indices], scores)
            except:
                model_group.attrs['auc'] = None
            model_group.attrs['train_time'] = train_time
            model_group.attrs['test_time'] = test_time
            model_group.create_dataset('test_scores', data=scores, compression=compression)
            # save whatever from the model
            model_config.storer.to_hdf5(model, model_group, compression=compression)
            # done
            model_group['DONE'] = 'Finished on %s' % strftime("%c")
        except Exception:
            model_group['FAILED'] = format_exc()
            if reraise:
                raise

    # Dataset hub
    dset = dset if isinstance(dset, ManysourcesDataset) else ManysourcesDataset(dset)
    # Features configuration
    feats_config = MANYSOURCES_FEATS[feats]
    molids, X, y = feats_config.extractor(dset)
    # Model configuration
    model_config = MANYSOURCES_MODELS[model]

    for expid in expids:
        print 'expid=%d' % expid
        lsocv = setup_splitters_lsocv(dset=dset, molids=molids, expids=(expid,))[0]
        h5_file = single_hdf5_per_exp(expid=expid, dset=dset.name, model=model, feats=feats)
        ensure_dir(op.dirname(h5_file))
        with h5py.File(h5_file) as h5:
            try:
                # coordinates
                try:
                    h5.require_group('dsets/%s' % dset.name).attrs['config'] = dset.name
                    h5.require_group('featss/%s' % feats).attrs['config'] = feats_config.name
                    h5.require_group('models/%s' % model).attrs['config'] = model_config_string(model_config.model)
                except:
                    pass  # Dodgy
                # dset coordinates
                dset_group = h5.require_group('/dset=%s' % dset.name)
                # features coordinates
                feat_group = dset_group.require_group('feats=%s' % feats)
                try:
                    feat_group.create_dataset('ids', data=molids, compression='lzf')
                except:
                    pass  # Data already there
                # expid group
                expid_group = feat_group.require_group('expid=%d' % expid)
                # LSO-CV
                lsocv_group = expid_group.require_group('lsocv')
                lsocv_group.attrs['config'] = lsocv.what().id()
                lsocv_group.attrs['num_folds'] = lsocv.num_folds
                lsocv_group.attrs['seed'] = lsocv.seed
                for split_num, splitter in enumerate(lsocv.splitters()):
                    generate_split_result(model_config, X, y,
                                          'fold=%d' % split_num, splitter, lsocv_group,
                                          reraise=reraise)
                # CRS-CV
                crscv = crscv_from_lsocv(lsocv)
                crscv_group = lsocv_group.require_group('crscv#seed=%d' % crscv.seed)
                crscv_group.attrs['config'] = crscv.what().id()
                crscv_group.attrs['num_folds'] = len(crscv.fold_sizes)
                crscv_group.attrs['seed'] = crscv.seed
                for split_num, splitter in enumerate(crscv.splitters()):
                    generate_split_result(model_config, X, y,
                                          'fold=%d' % split_num, splitter,
                                          crscv_group, reraise=reraise)
            except:
                if 'TOP_FAIL' not in h5:
                    h5['TOP_FAIL'] = format_exc()
                if reraise:
                    raise

    print 'DONE'


def check_regenerate_results(regenerate=False):
    for expid, dset, feat, model in product(range(4096),
                                            MANYSOURCES_MOLECULES.keys(),
                                            MANYSOURCES_FEATS.keys(),
                                            MANYSOURCES_MODELS.keys()):
        h5_file = single_hdf5_per_exp(expid=expid, dset=dset, feats=feat, model=model)
        if not op.isfile(h5_file):
            print 'Missing %s ' % h5_file
            if regenerate:
                try:
                    generate_lsocv_results(dset=dset, model=model, feats=feat, expids=(expid,), reraise=True)
                except:
                    print format_exc()
                    pass
                if not op.isfile(h5_file):
                    print '\tRegeneration failed'
                else:
                    print '\tRegeneration successful'
    # Still some will fail but the HDF5 file will be there FIXME: test for that


def lsocvs_cls(step=32,
               total=4096,
               destinies=(('galileo', 46),
                          ('zeus',    32),
                          ('str22',   10),
                          ('strz',    22)),
               batch_name='logregs1',
               root=op.abspath(op.join(op.dirname(__file__), '..')),
               feats_models=(('ecfps1', 'logreg1'),
                             ('ecfps1', 'logreg3'),
                             ('ecfps1', 'logreg5'),
                             ('ecfps1', 'logreg7'),
                             ('rdkdescs1', 'rfc1'))):
    MANYSOURCES_LOGS_DIR = op.join('~', 'manysources-logs', batch_name)
    commands = []
    for dset, (feats, model) in product(MANYSOURCES_MOLECULES.keys(), feats_models):
        for start in xrange(step):
            expids = range(start, total, step)
            logfile = op.join(MANYSOURCES_LOGS_DIR,
                              '%s-%s-%s-%d_%d_%d.log' % (dset, feats, model, step, start, total))
            commands.append(
                'PYTHONPATH=.:$PYTHONPATH python2 -u manysources/experiments.py generate-lsocv-results '
                '--dset %s --feats %s --model %s --expids %s '
                '&>%s' %
                (dset, feats, model, ' '.join(map(str, expids)), logfile))

    # --- Save the cls to files

    # Remove duplicates, randomize
    commands = list(set(commands))

    # Proper balance of workloads between machines
    destinies = [(name, ['mkdir -p %s' % MANYSOURCES_LOGS_DIR], prob) for name, prob in destinies]
    p_choice = np.array([p for _, _, p in destinies], dtype=np.float)
    p_choice /= float(np.sum(p_choice))
    rng = np.random.RandomState(2147483647)
    for cl in commands:
        _, your_destiny, _ = destinies[rng.choice(len(destinies), p=p_choice)]
        your_destiny.append(cl)

    # Save the selections
    for name, cls, _ in destinies:
        with open(op.join(root, name + '-' + batch_name), 'w') as writer:
            writer.write('\n'.join(cls))

    # ----- Summary
    total_cls = sum(len(cl) for _, cl, _ in destinies)
    print 'Total number of commands: %d' % total_cls
    for name, cls, p in destinies:
        print '\t%s\t%d %g %g' % (name.ljust(30), len(cls), p, len(cls) / (total_cls + 1.))


# --- Result read logic

class ManysourcesResult(object):

    def __init__(self, expid=0, dset='bcrp', feats='ecfps1', model='logreg1'):
        super(ManysourcesResult, self).__init__()
        self.expid = expid
        self.dset = dset
        self.model = model
        self.feats = feats

        # Caches
        self._h5_handle = None
        self._molids = None
        self._msdsest = None

    def h5_file(self):
        sfe = single_hdf5_per_exp(expid=self.expid,
                                  model=self.model,
                                  dset=self.dset,
                                  feats=self.feats)
        return sfe if op.exists(sfe) else merged_hdf5_file(expid=self.expid,
                                                           model=self.model,
                                                           dset=self.dset,
                                                           feats=self.feats)
    # FIXME: check that actually the result is in the HDF5 file

    def _h5(self):
        if self._h5_handle is None:
            self._h5_handle = h5py.File(self.h5_file(), 'r')
        return self._h5_handle

    def _close_h5(self):
        if self._h5_handle is not None:
            try:
                self._h5_handle.close()
            finally:
                self._h5_handle = None
        # Put in destructor and context manager

    def __enter__(self):
        self._h5()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._close_h5()

    def __del__(self):
        self.__exit__(None, None, None)

    def _h5_group_or_none(self, group):
        try:
            return self._h5()[group]
        except:
            return None

    def _h5_dset_group(self):
        return self._h5_group_or_none('/dsets/%s' % self.dset)

    def dset_config(self):
        return self._h5_dset_group().attrs['config']

    def _h5_feats_group(self):
        return self._h5_group_or_none('/models/%s' % self.model)

    def feats_config(self):
        return self._h5_model_group().attrs['config']

    def _h5_model_group(self):
        return self._h5_group_or_none('/models/%s' % self.model)

    def model_config(self):
        return self._h5_model_group().attrs['config']

    def _h5_dset_result_group(self):
        return self._h5_group_or_none('/dset=%s' % self.dset)

    def _h5_feats_result_group(self):
        return self._h5_group_or_none('/dset=%s/feats=%s' % (self.dset, self.feats))

    def molids(self, cache=True):
        if cache:
            if self._molids is None:
                self._molids = self.molids(cache=False)
            return self._molids
        return self._h5_feats_result_group()['ids'][:]

    def ids(self):
        return self.molids()

    def num_examples(self):
        return len(self.ids())

    def num_molecules(self):
        return self.num_examples()

    def _h5_expid_group(self):
        return self._h5_group_or_none('/dset=%s/feats=%s/expid=%d' % (self.dset, self.feats, self.expid))

    def _h5_lsocv_group(self):
        return self._h5_expid_group()['lsocv']

    def lsocv(self):
        return ManySourcesCVResult(self, self._h5_lsocv_group())

    def crscv(self, seed=None):
        if seed is None:
            seed = self.expid
        return ManySourcesCVResult(self, self._h5_lsocv_group()['crscv#seed=%d' % seed])

    def ms_dset(self):
        if self._msdsest is None:
            self._msdsest = ManysourcesDataset(self.dset)
        return self._msdsest


class TrainTestSplitResult(object):

    def __init__(self, parent, splitgroup, model_config=None):
        super(TrainTestSplitResult, self).__init__()
        self.parent = parent
        self.splitgroup = splitgroup
        self._model_config = model_config if model_config is not None else MANYSOURCES_MODELS[self.model()]

    def config(self):
        return self.splitgroup.attrs['config']

    def test_indices(self):
        return self.splitgroup['test_indices'][:]

    def y_test(self):
        return self.splitgroup['y_test'][:]  # Maybe redundant

    def pos_proportion(self, pos=1):
        y_test = self.y_test()
        pos_count = float(np.sum(self.y_test() == pos))
        return pos_count / len(y_test)

    def ids(self):
        return self.parent.molids()

    def molids(self):
        return self.ids()

    def num_examples(self):
        return len(self.ids())

    def model(self):
        return self.parent.model()

    def _h5_model_group(self):
        return self.splitgroup['model=%s' % self.model()]

    def auc(self):
        try:
            return self._h5_model_group().attrs['auc']
        except:
            return np.nan

    def test_size(self):
        return len(self.test_indices())

    def train_time(self):
        return self._h5_model_group().attrs['train_time']

    def test_time(self):
        return self._h5_model_group().attrs['test_time']

    def test_scores(self):
        return self._h5_model_group()['test_scores'][:]

    def model_data(self):
        return self._model_config.storer.from_hdf5(self._h5_model_group())

    def coocurrences(self):
        coocurrences = np.zeros(self.num_examples(), np.bool)
        coocurrences[self.test_indices()] = True
        return coocurrences

    def ms_dset(self):
        return self.parent.ms_dset()

    def sources_in_test(self, dset=None):
        if dset is None:
            dset = self.ms_dset()
        molids = self.molids()[self.test_indices()]
        return sorted(set(dset.mols().molids2sources(molids)))

    def sources_coocurrences(self, dset=None):
        if dset is None:
            dset = self.ms_dset()
        sources_in_test = self.sources_in_test(dset=dset)
        x = np.zeros(dset.mols().num_sources(), dtype=np.bool)
        x[dset.mols().sources2is(sources_in_test)] = True
        return x


class ManySourcesCVResult(object):

    def __init__(self, parent_result, cvgroup):
        super(ManySourcesCVResult, self).__init__()
        self.parent = parent_result
        self.cvgroup = cvgroup
        # Caches
        self._folds = None

    def config(self):
        return self.cvgroup.attrs['config']

    def num_folds(self):
        return self.cvgroup.attrs['num_folds']

    def present_fold_ids(self):
        if self._folds is None:
            folds = [k for k in self.cvgroup.keys() if k.startswith('fold=')]
            self._folds = sorted(folds, key=lambda fold_name: int(fold_name[len('fold='):]))
        return self._folds

    def seed(self):
        return self.cvgroup.attrs['seed']

    def molids(self):
        return self.parent.molids()

    def ids(self):
        return self.molids()

    def num_examples(self):
        return len(self.molids())

    def model(self):
        return self.parent.model

    def fold(self, fold_id):
        if isinstance(fold_id, int):
            fold_id = 'fold=%d' % fold_id
        return TrainTestSplitResult(self, self.cvgroup[fold_id])

    def folds(self):
        return (self.fold(fold_id) for fold_id in self.present_fold_ids())

    def fold_size(self, fold_id):
        return len(self.fold(fold_id).test_indices())

    def fold_sizes(self):
        return map(self.fold_size, self.present_fold_ids())

    def all_aucs(self):
        return np.array([fold.auc() for fold in self.folds()])

    def mean_auc(self):
        aucs = self.all_aucs()
        return np.mean(aucs), np.std(aucs)

    def ms_dset(self):
        return self.parent.ms_dset()

    def merge_scores(self, calibration=None):
        """
        Merges all the scores in the CV so that we generate a new feature.

        Parameters
        ----------
        calibration: string, default None
            Several strategies are available for calibrating the scores, so that
              - None: scores are left as they are (so probably scale issues will happen here and there)
              - 'max-cheating': calibrate on the actual labels of the scores to calibrate;
                                this is brutal test-leakage probably leading to performance overstimation
              - 'other-folds': calibrate using the data from other folds to build the calibrator
                               still leakage present, but funnily it can lead to lower performance estimates
              - 'all-scores': calibrate using all the scores and all the labels to fit the calibrator
                              this is less outrageous and can help when merging with the results of other CVs
                              (for example it won't change ROC, but can change individual errors
                               and overfit if combined with other features)
              - '0-1': each fold predictions get calibrated between 0 and 1
              - 'std-log': each fold predictions get standardized and then passed through a logistic func
            N.B. what would work, or better say cheat less, is to include a calibrator in model building,
                 training on holdouts and then passing the final scores by these calibrators;
                 implement if needed / time allowing
        Returns
        -------
        a tuple (all_scores, all_ys, all_folds) containing, respectively and per exaple,
        the example score, its actual label and the fold number in which the score was obtained,
        """
        all_folds = np.empty(self.num_examples(), dtype=np.int)
        all_scores = np.empty(self.num_examples())
        all_ys = np.empty(self.num_examples(), dtype=np.int)

        # No calibration / "all" calibration
        if calibration is None or 'all' in calibration:
            for fold_num, fold in enumerate(self.folds()):
                test_indices = fold.test_indices()
                all_scores[test_indices] = fold.test_scores()
                all_ys[test_indices] = fold.y_test()
                all_folds[test_indices] = fold_num
            if calibration is not None:
                calibrator = LogisticRegression()
                calibrator.fit(all_scores.reshape(-1, 1), all_ys)
                all_scores = calibrator.predict_proba(all_scores.reshape(-1, 1))[:, 1]
        # "Max cheating" calibration (calibrate each fold with its actual labels)
        elif 'max' in calibration:
            for fold_num, fold in enumerate(self.folds()):
                test_indices = fold.test_indices()
                test_scores = fold.test_scores()
                y_test = fold.y_test()
                all_ys[test_indices] = y_test
                all_folds[test_indices] = fold_num
                calibrator = LogisticRegression()
                calibrator.fit(test_scores.reshape(-1, 1), y_test)
                all_scores[test_indices] = calibrator.predict_proba(test_scores.reshape(-1, 1))[:, 1]
        # "Other fold calibration (calibrate each fold using other folds score,y pairs, little better than max-cheating)
        elif 'other' in calibration:
            prefetched_scores = [(fold_num, fold.test_scores(), fold.test_indices(), fold.y_test())
                                 for fold_num, fold in enumerate(self.folds())]
            for fold_num, test_scores, test_indices, y_test in prefetched_scores:
                all_folds[test_indices] = fold_num
                all_ys[test_indices] = y_test
                other_scores = np.hstack([scores for _, scores, _, _ in prefetched_scores]).reshape(-1, 1)
                other_ys = np.hstack([y for _, _, _, y in prefetched_scores])
                calibrator = LogisticRegression()
                calibrator.fit(other_scores, other_ys)
                all_scores[test_indices] = calibrator.predict_proba(test_scores.reshape(-1, 1))[:, 1]
        elif '0-1' in calibration:
            for fold_num, fold in enumerate(self.folds()):
                test_indices = fold.test_indices()
                ts = fold.test_scores()
                if ts.max() > ts.min():
                    all_scores[test_indices] = (ts - ts.min()) / (ts.max() - ts.min())
                else:
                    all_scores[test_indices] = ts  # constant vector
                all_ys[test_indices] = fold.y_test()
                all_folds[test_indices] = fold_num
        elif 'std-log' in calibration:
            for fold_num, fold in enumerate(self.folds()):
                test_indices = fold.test_indices()
                ts = fold.test_scores()
                if ts.std() != 0:
                    ts = 1.0 / (1.0 + np.exp(-0.5 * (ts - ts.mean()) / ts.std()))
                    all_scores[test_indices] = ts
                else:
                    all_scores[test_indices] = ts  # constant vector
                all_ys[test_indices] = fold.y_test()
                all_folds[test_indices] = fold_num
        else:
            raise Exception('%s calibration strategy unknown' % calibration)

        return all_scores, all_ys, all_folds

    def coocurrence_matrix(self):
        """Returns a coocurrence matrix (numfolds x numexamples).
        Each row is a fold, each column is 1 iff the corresponding example was in test, otherwise it is 0.
        """
        return np.array([fold.coocurrences() for fold in self.folds()])

    def sources_coocurrence_matrix(self):
        return np.array([fold.sources_coocurrences() for fold in self.folds()])

    def sources_coocurrence_matrix_only_folds_with_source(self, source_index):  # FIXME: support source_name too
                                                                                # Beautiful name function
        return [row for row in self.sources_coocurrence_matrix() if row[source_index]]


def next_n(n, size=4000):
    return range(n * size, n * size + size)

# These are the experiments for our dear l1-logreg and ecfps1

# generate_lsocv_results(dset='bcrp', model='logreg3', feats='ecfps1', expids=next_n(9), reraise=False)
# generate_lsocv_results(dset='hERG', model='logreg3', feats='ecfps1', expids=next_n(0), reraise=False)
# generate_lsocv_results(dset='mutagenicity', model='logreg3', feats='ecfps1', expids=next_n(0), reraise=False)
# generate_lsocv_results(dset='pgp-barbara', model='logreg3', feats='ecfps1', expids=next_n(0), reraise=False)
# generate_lsocv_results(dset='pgp-barbara-verapamil', model='logreg3', feats='ecfps1', expids=next_n(0), reraise=False)
# generate_lsocv_results(dset='pgp-cruciani', model='logreg3', feats='ecfps1', expids=next_n(0), reraise=False)


def packres():
    """
    By-hand packing of results
    """
    root = op.join(home, 'data-analysis', 'chem', 'manysources', 'results', 'bcrp')
    expids = glob(op.join(root, 'expid=*'))
    expids = sorted(expids, key=lambda x: int(op.basename(x).partition('expid=')[2]))
    print expids

# packres()
# exit(22)


if __name__ == '__main__':

    import argh

    parser = argh.ArghParser()
    parser.add_commands((
        lsocvs_cls,
        generate_lsocv_results,
        check_regenerate_results,
    ))
    parser.dispatch()

####################################################################################
#
# MISSING: Global CVs (should be computed and stored on the features level)
# MISSING: Inner CVs (let's create a separate function for them, so that we keep manageable the main one)
# Not so important here...
#
####################################################################################
#
# TODO: check compressibility of different models (stat the dataset in the HDF5 for LZF?)
#
####################################################################################
#
# TODO: merge HDF5s into other big ones; for example, using h5copy
#       http://stackoverflow.com/questions/18492273/combining-hdf5-files
#       http://stackoverflow.com/questions/5346589/concatenate-a-large-number-of-hdf5-files
# Or use HDF5 mount
#       http://davis.lbl.gov/Manuals/HDF5-1.4.3/Tutor/mount.html
#       https://groups.google.com/forum/#!topic/h5py/PrMVy-URec8
#
####################################################################################
#
# TODO: use HDF5 symlinks can help grouping stuff by different thingys
#
####################################################################################
#
# TODO: Attention at the performance on random forests for small training sets (e.g. bcrp without Matsson)
#
####################################################################################
