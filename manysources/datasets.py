# coding=utf-8
"""Convenience access to different aspects of concrete datasets of interest."""
import os.path as op
import h5py
import numpy as np
from joblib import Parallel, delayed, cpu_count
from manysources import MANYSOURCES_ROOT
from manysources.common.misc import ensure_dir
from manysources.common.molecules import MoleculesFromRDKit, string2binary
from manysources.common.rdkit_utils import RDKitDescripter
from manysources.features import RDKMorganFingerprinter, UnfoldedFingerprints, zero_dupes

# --- Data coordinates on hard-disk

MANYSOURCES_DATA_ROOT = op.join(MANYSOURCES_ROOT, 'data')


def manysources_dataset_root(dset):
    return op.join(MANYSOURCES_DATA_ROOT, dset)


def manysources_sdf(dset):
    return op.join(manysources_dataset_root(dset), '00-sdf', '%s.sdf' % dset)


def manysources_results_root(dset):
    return op.join(MANYSOURCES_DATA_ROOT, 'results', dset)


# --- BCRP


BCRP_SDF = manysources_sdf('bcrp')
BCRP_NUM_EXPECTED_MOLS = 978


def bcrp_molecules(sdf=BCRP_SDF):
    return MoleculesFromRDKit(sdf=sdf,
                              dataset_id='BCRP',
                              label_extractor='Activity',
                              source_extractor='ActivitySource',
                              label2target=string2binary('INHIBITOR'),
                              num_expected_mols=BCRP_NUM_EXPECTED_MOLS)


# --- hERGmeout

HERG_SDF = manysources_sdf('hERG')
HERG_NUM_EXPECTED_MOLS = 2844


def herg_molecules(sdf=HERG_SDF):
    return MoleculesFromRDKit(sdf=sdf,
                              dataset_id='hERGmeout',
                              label_extractor='Activity',
                              source_extractor='Source',
                              label2target=string2binary('1'),
                              num_expected_mols=HERG_NUM_EXPECTED_MOLS)

# --- hERG-MLSMR

HERG_MLSMR = manysources_sdf('hERG-mlsmr')
HERG_MLSMR_NUM_EXPECTED_MOLS = None


def herg_mlsmr_molecules(sdf=HERG_MLSMR):
    return MoleculesFromRDKit(sdf=sdf,
                              dataset_id='hERG-mlsmr',
                              label_extractor='Activity',
                              source_extractor='Source',
                              label2target=string2binary('1'),
                              num_expected_mols=HERG_MLSMR_NUM_EXPECTED_MOLS)

# --- Mutagenicity


MUTAGENICITY_SDF = manysources_sdf('mutagenicity')
MUTAGENICITY_NUM_EXPECTED_MOLS = 6352
# Why not 6364?
#  Flo -> I have passed some useless programs that
#   - have failed for some random molecules
#   - and screwed up some other molecules on the way that I do not know


def mutagenicity_molecules(sdf=MUTAGENICITY_SDF):
    return MoleculesFromRDKit(sdf=sdf,
                              dataset_id='Mutagenicity',
                              label_extractor='Activity',
                              source_extractor='Source',
                              label2target=string2binary('1'),
                              num_expected_mols=MUTAGENICITY_NUM_EXPECTED_MOLS)

# --- PGP-Barbara

PGPBARBARA_SDF = manysources_sdf('pgp-barbara')
PGPBARBARA_NUM_EXPECTED_MOLS = 407
# Why not 408?
# Flo -> I have passed some useless programs that
#   - have failed for some random molecules
#   - and screwed up some other molecules on the way that I do not know


def pgpbarbara_molecules(sdf=PGPBARBARA_SDF):
    return MoleculesFromRDKit(sdf=sdf,
                              dataset_id='PgPBarbara',
                              label_extractor='Activity',
                              source_extractor='Source',
                              label2target=string2binary('1'),
                              num_expected_mols=PGPBARBARA_NUM_EXPECTED_MOLS)


# --- PGP-Barbara-Verapamil

PGPBARBARAVERAPAMIL_SDF = manysources_sdf('pgp-barbara-verapamil')
PGPBARBARAVERAPAMIL_NUM_EXPECTED_MOLS = 388  # Why not 407?


def pgpbarbaraverapamil_molecules(sdf=PGPBARBARAVERAPAMIL_SDF):
    return MoleculesFromRDKit(sdf=sdf,
                              dataset_id='PgPBarbaraVerapamil',
                              label_extractor='Activity',
                              source_extractor='Source',
                              label2target=string2binary('1'),
                              num_expected_mols=PGPBARBARAVERAPAMIL_NUM_EXPECTED_MOLS)


# --- PGP-Cruciani

PGPCRUCIANI_SDF = manysources_sdf('pgp-cruciani')
PGPCRUCIANI_NUM_EXPECTED_MOLS = 1180


def pgpcruciani_molecules(sdf=PGPCRUCIANI_SDF):
    return MoleculesFromRDKit(sdf=sdf,
                              dataset_id='PgPCruciani',
                              label_extractor='Activity',
                              source_extractor='Source',
                              label2target=string2binary('1'),
                              num_expected_mols=PGPCRUCIANI_NUM_EXPECTED_MOLS)


# --- Convenient aggregation of experiments data

MANYSOURCES_MOLECULES = {
    'bcrp': bcrp_molecules,
    'hERG': herg_molecules,
    'mutagenicity': mutagenicity_molecules,
    'pgp-barbara': pgpbarbara_molecules,
    'pgp-barbara-verapamil': pgpbarbaraverapamil_molecules,
    'pgp-cruciani': pgpcruciani_molecules,
}

MANYSOURCES_LSO_NUMCVFOLDS = {
    'bcrp':                  (3, 4, 5, 6, 8, 10),
    'hERG':                  (3, 4, 5, 6, 8, 10),
    'mutagenicity':          (3, 4, 5, 6, 8, 10),  # Original code, 12 to create "small externals"
    'pgp-barbara':           (3, 4, 5, 6, 8, 10),
    'pgp-barbara-verapamil': (3, 4, 5, 6, 8, 10),
    'pgp-cruciani':          (3, 4, 5, 6, 8, 10),
}

MANYSOURCES_LSO_PROPORTIONS = {
    'bcrp':                  (1./3, 1./4, 1./5, 1./6, 1./8, 1./10),
    'hERG':                  (1./3, 1./4, 1./5, 1./6, 1./8, 1./10),
    'mutagenicity':          (1./3, 1./4, 1./5, 1./6, 1./8, 1./10),  # Original code, 1./12 to create "small externals"
    'pgp-barbara':           (1./3, 1./4, 1./5, 1./6, 1./8, 1./10),
    'pgp-barbara-verapamil': (1./3, 1./4, 1./5, 1./6, 1./8, 1./10),
    'pgp-cruciani':          (1./3, 1./4, 1./5, 1./6, 1./8, 1./10),
}


class ManysourcesDataset(object):

    def __init__(self, dset='bcrp'):
        super(ManysourcesDataset, self).__init__()
        self.name = dset
        self._molecules = None
        # RDK ECFPs
        self._ecfps = None
        self._ecfps_no_dupes = None
        # RDK Descriptors
        self._rdkdescs = None
        self._rdkdescs_molids = None
        self._rdkdescs_fnames = None

    def mols(self):
        if self._molecules is None:
            pickle_file = op.join(manysources_dataset_root(self.name), '01-molspickle', '%s.pickled.gz' % self.name)
            if not op.isfile(pickle_file):
                self._molecules = MANYSOURCES_MOLECULES.get(self.name, lambda: None)()
                ensure_dir(op.dirname(pickle_file))
                self._molecules.to_pickle(pickle_file, compressed=True)
                with open(op.join(op.dirname(pickle_file), 'config.json'), 'w') as writer:
                    writer.write(self._molecules.what().id())
            else:
                self._molecules = MoleculesFromRDKit.from_pickle(pickle_file, compressed=True)
        return self._molecules

    def molecules(self):
        return self.mols()

    # --- Fingerprints

    def _unfolded_ecfps(self):
        if self._ecfps is None:
            ecfp_file = op.join(manysources_dataset_root(self.name), '02-ecfps', '%s.ecfps.h5' % self.name)
            if not op.exists(ecfp_file):
                fingerprinter = RDKMorganFingerprinter()
                for molid, mol in self.mols():
                    print molid
                    fingerprinter.add_mol(molid, mol)
                ensure_dir(op.dirname(ecfp_file))
                with open(op.join(op.dirname(ecfp_file), 'config.json'), 'w') as writer:
                    writer.write(fingerprinter.what().id())
                fingerprinter.fingerprints().save(ecfp_file)
            self._ecfps = UnfoldedFingerprints.load(ecfp_file)
        return self._ecfps

    def _unfolded_ecfps_nodupes(self):
        if self._ecfps_no_dupes is None:
            ecfp_nodupes_file = op.join(manysources_dataset_root(self.name),
                                        '03-ecfps-nodupes', '%s.ecfps.h5' % self.name)
            if not op.exists(ecfp_nodupes_file):
                ecfps = self._unfolded_ecfps()
                ufp = UnfoldedFingerprints(ecfps.molids,
                                           ecfps.i2s,
                                           zero_dupes(ecfps.csr, by_rows=False),  # Hack 1, this should be Configurable
                                           failed_molids=ecfps.failed_molids)
                ensure_dir(op.dirname(ecfp_nodupes_file))
                with open(op.join(op.dirname(ecfp_nodupes_file), 'config.txt'), 'w') as writer:
                    writer.write('Same as 02-ecfps, but removed columns that have the same value accross all rows.')
                ufp.save(ecfp_nodupes_file)
            self._ecfps = UnfoldedFingerprints.load(ecfp_nodupes_file)
        return self._ecfps

    def ecfps(self, no_dupes=False):
        return self._unfolded_ecfps_nodupes() if no_dupes else self._unfolded_ecfps()

    def ecfpsXY(self, no_dupes=False):
        ufp = self.ecfps(no_dupes=no_dupes)
        return ufp.csr, self.Y(ufp.molids)

    def ecfps_molidsXY(self, no_dupes=False):
        ufp = self.ecfps(no_dupes=no_dupes)
        return ufp.molids, ufp.csr, self.Y(ufp.molids)

    # --- RDKitDescriptors

    def rdkdescs(self, keep_with_missing=False):
        if self._rdkdescs is None:
            rdkdescs_file = op.join(manysources_dataset_root(self.name), '02-rdkdescs', '%s.rdkdescs.h5' % self.name)
            if not op.exists(rdkdescs_file):
                computer = RDKitDescripter()
                descs = computer.compute(self.mols().mols())
                molids = self.mols().molids()  # We assume all mols have descriptors computed, even if missing
                fnames = computer.fnames()
                ensure_dir(op.dirname(rdkdescs_file))
                with h5py.File(rdkdescs_file, 'w-') as h5:
                    h5['rdkdescs'] = descs
                    h5['molids'] = molids
                    h5['fnames'] = fnames
                with open(op.join(op.dirname(rdkdescs_file), 'config.json'), 'w') as writer:
                    writer.write(computer.what().id())
            with h5py.File(rdkdescs_file, 'r') as h5:
                self._rdkdescs = h5['rdkdescs'][:]
                self._rdkdescs_molids = h5['molids'][:]
                self._rdkdescs_fnames = h5['fnames'][:]
                if not keep_with_missing:
                    to_keep = np.all(np.isfinite(self._rdkdescs), axis=1)
                    self._rdkdescs = self._rdkdescs[to_keep, :]
                    self._rdkdescs_molids = self._rdkdescs_molids[to_keep]

        return self._rdkdescs_molids, \
            self._rdkdescs_fnames, \
            self._rdkdescs, \
            self.Y(self._rdkdescs_molids)

    # --- Target

    def Y(self, molids=None):
        if molids is None:
            molids = self.mols().molids()
        return np.array([self.mols().molid2target(molid) for molid in molids])

    # --- LSO Experimental Setup

    def lso_proportions(self):
        return MANYSOURCES_LSO_PROPORTIONS[self.name]

    def lso_numcvfolds(self):
        return MANYSOURCES_LSO_NUMCVFOLDS[self.name]

    # --- Others

    def molecules_summary(self):
        print 'There are %d molecules in %d sources.' % (self.mols().num_mols(), self.mols().num_sources())


def cache_dset(dset):
    dset = ManysourcesDataset(dset)
    dset.molecules()
    dset.ecfps(no_dupes=False)
    dset.ecfps(no_dupes=True)  # TODO: Better tracking
    dset.rdkdescs()            # TODO: better modularity, preprocessing on demand (missings and normalisation)


if __name__ == '__main__':

    Parallel(n_jobs=cpu_count())(delayed(cache_dset)(dset)
                                 for dset in MANYSOURCES_MOLECULES.keys())

    print('Done')
    exit(0)
