# coding=utf-8
import gzip
import cPickle as pickle

import numpy as np
from whatami import whatable

from manysources.common.rdkit_utils import iterate_rdkit_mols_in_sdf_file


@whatable
class Molecules(object):
    """A collection of molecules, identified by unique ids and associated with some attributes."""

    def __init__(self):
        super(Molecules, self).__init__()
        # Some ad-hoc caches
        self._i2source = None
        self._source2i = None

    def molids(self):
        """Returns a list of molids present in the dataset.
           If the molid cannot be found in the dataset, raises an Exception.
        """
        raise NotImplementedError('To implement, to ABC')

    def num_mols(self):
        return len(self.molids())

    def molid2mol(self, molid):
        """Returns a molecule object (can be rdkit.rdMol, pybel,Mol... whatever the backend dictates)."""
        raise NotImplementedError('To implement, to ABC')

    def mols(self, as_list=True):
        if as_list:
            return [self.molid2mol(molid) for molid in self.molids()]
        return (self.molid2mol(molid) for molid in self.molids())

    def molid2source(self, molid):
        """Returns the source id for the molecule.
           If the molid cannot be found in the dataset, raises an Exception.
        """
        raise NotImplementedError('To implement, to ABC')

    def molids2sources(self, molids=None, as_list=True):
        molids = molids if molids is not None else self.molids()
        if as_list:
            return [self.molid2source(molid) for molid in molids]
        return (self.molid2source(molid) for molid in molids)

    def present_sources(self):
        return set(self.molids2sources(as_list=False))

    def num_sources(self):
        return len(self.present_sources())

    def source2molids(self, source):
        return [molid for molid in self.molids() if self.molid2source(molid) == source]

    def sources2molids(self, sources):
        sources = set(sources)
        return sorted([molid for molid in self.molids() if self.molid2source(molid) in sources])

    def i2source(self, i):
        if not hasattr(self, '_i2source') or self._i2source is None:  # Pickles missing these attrs, ergo hasattr FIXME
            self._i2source = sorted(self.present_sources())
        return self._i2source[i]

    def source2i(self, source):
        if not hasattr(self, '_source2i') or self._source2i is None:  # Pickles missing these attrs, ergo hasattr FIXME
            self.i2source(0)  # FIXME: bug if no mols, hack to work until the pickle problem is solved
            self._source2i = {s: i for i, s in enumerate(self._i2source)}
        return self._source2i[source]

    def sources2is(self, sources):
        return map(self.source2i, sources)

    def is2sources(self, i_s):
        return map(self.i2source, i_s)

    def i2sources_order(self):
        if not hasattr(self, '_i2source'):
            self.i2source(0)
        return np.array(self._i2source)

    def molid2source_dict(self):
        return {molid: self.molid2source(molid) for molid in self.molids()}

    def molid2label(self, molid):
        """Returns the label associated with a molid.
        If the molid cannot be found in the dataset, raises an Exception.
        """
        raise NotImplementedError('To implement, to ABC')

    def labels(self, as_list=True):
        if as_list:
            return [self.molid2label(molid) for molid in self.molids()]
        return (self.molid2label(molid) for molid in self.molids())

    def molid2label_dict(self):
        return {molid: self.molid2label(molid) for molid in self.molids()}

    def molid2target(self, molid):
        """Returns the target associated with a molid.
        If the molid cannot be found in the dataset, raises an Exception.
        """
        raise NotImplementedError('To implement, to ABC')

    def targets(self, as_list=True):
        if as_list:
            return [self.molid2target(molid) for molid in self.molids()]
        return (self.molid2target(molid) for molid in self.molids())

    def molid2target_dict(self):
        return {molid: self.molid2target(molid) for molid in self.molids()}

    def __iter__(self):
        return ((molid, self.molid2mol(molid)) for molid in self.molids())


class MoleculesFromRDKit(Molecules):

    def __init__(self,
                 sdf,
                 sanitize=True,
                 dataset_id=None,
                 label_extractor=lambda label_string: float(label_string),  # Regression
                 label2target=None,
                 molid_extractor=None,
                 source_extractor=None,
                 alpha_order=False,
                 num_expected_mols=None):
        """Reads in memory the whole dataset.
        For the moment we only accept sdf, it is trivial to accept other sources.
        """
        super(MoleculesFromRDKit, self).__init__()

        # Here we use rdkit, but we want to limit the scope of the dependency
        from rdkit.Chem import PropertyMol

        # Provenance info
        self._sdf = sdf
        self.dataset_id = dataset_id
        self.sanitize = sanitize

        # --- Relevant molecule field extractors

        def field_extractor(extractor, default, extractor_name):
            if extractor is None:
                extractor = lambda rdkit_mol: rdkit_mol.GetProp(default)
            elif isinstance(extractor, str):
                field_id = extractor
                extractor = lambda rdkit_mol: rdkit_mol.GetProp(field_id)
            elif not callable(extractor):  # TODO: also check that it has the correct signature
                raise Exception('The %s extractor must be either a field name '
                                'on the molecule dictionary or a function' % extractor_name)
            return extractor
        # ID extractor
        molid_extractor = field_extractor(molid_extractor, '_Name', 'id')
        # Label extractor
        label_extractor = field_extractor(label_extractor, 'Activity', 'label')
        # Source extractor
        source_extractor = field_extractor(source_extractor, 'Source', 'source')

        # Function to put the label into something that can be predicted (e.g. {0, 1})
        if not callable(label2target):
            raise Exception('label2target needs to be callable')

        # --- Populate the object with the data

        self._molinfos = {}
        self._molids = []
        self._sources = set()
        seen_mols = set()

        # By using a custom iterator we can detect formatting issues
        # Actually SDMolSupplier also can be queried about the text it is trying to parse
        mol_iterator = iterate_rdkit_mols_in_sdf_file(self._sdf, sanitize=sanitize)

        for molnum, (mol, molblock) in enumerate(mol_iterator):
            if mol is None:
                raise Exception('Problem reading molecule %d, aborting' % molnum)
            mol = PropertyMol.PropertyMol(mol)
            molid = molid_extractor(mol)
            if molid in seen_mols:
                raise Exception('Duplicate molids are not supported, but %s is there more than once' % molid)
            source = source_extractor(mol)
            label = label_extractor(mol)
            target = label2target(label)
            self._sources.add(source)
            self._molinfos[molid] = (mol, source, label, target)
            self._molids.append(molid)

        # --- Sort molids, make them immutable

        if alpha_order:
            self._molids = sorted(self._molids)
        self._molids = tuple(self._molids)

        # --- Sanity-check the number of expected molecules

        if num_expected_mols is not None and num_expected_mols != self.num_mols():
            raise Exception('Molecules %s.\n\tExpected %d molecules, but there are %d.' % (self.dataset_id,
                                                                                           num_expected_mols,
                                                                                           self.num_mols()))

    def molids(self):
        return self._molids

    def molid2mol(self, molid):
        return self._molinfos[molid][0]

    def molid2source(self, molid):
        return self._molinfos[molid][1]

    def molid2label(self, molid):
        return self._molinfos[molid][2]

    def molid2target(self, molid):
        return self._molinfos[molid][3]

    def sdf(self):
        return self._sdf

    def to_pickle(self, path, compressed=True):
        with gzip.open(path, 'w') if compressed else open(path, 'w') as writer:
            pickle.dump(self, writer, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def from_pickle(path, compressed=True):
        with gzip.open(path, 'r') if compressed else open(path, 'r') as reader:
            return pickle.load(reader)


# --- Helper functions

def string2binary(target_class_label='Active'):
    """Returns a function mapping a string str to 0 if str is not "target_class_label" and 1 otherwise.
    If target_class_label is None, returns None (missing class).
    """
    return lambda label: None if target_class_label == 'None' else 0 if target_class_label != label else 1


################################
# This is for documenting the contracts and for Flo explanation
# TODO: ready for streaming
################################
