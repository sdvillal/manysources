# coding=utf-8
from array import array
from collections import defaultdict
import hashlib
from itertools import chain
from operator import itemgetter
import collections

import h5py
from scipy.sparse import coo_matrix, csr_matrix
import numpy as np
from whatami import whatable

from manysources import warning
from manysources.common.rdkit_utils import unfolded_morgan_fingerprint


# --- Unfolded Morgan Fingerprints (via rdkit)
#     Heavily based on our code for malaria, but with the easiness of small datasets

class UnfoldedFingerprints(object):

    def __init__(self, molids, i2s, csr, failed_molids=None):
        super(UnfoldedFingerprints, self).__init__()
        self.i2s = i2s
        self.s2i = {s: i for i, s in enumerate(i2s)}
        self.csr = csr
        self.molids = molids
        self.failed_molids = failed_molids

    def save(self, h5):  # TODO: this is generic to all unfolded fingerprints, extract
        with h5py.File(h5, 'w-') as h5:
            h5['molids'] = self.molids
            h5['failed_molids'] = self.failed_molids
            h5['i2s'] = self.i2s  # Will be inefficient?
            h5['data'] = self.csr.data.astype(np.int32)
            h5['indices'] = self.csr.indices.astype(np.int32)
            h5['indptr'] = self.csr.indptr.astype(np.int32)
            h5['shape'] = self.csr.shape

    @staticmethod
    def load(h5):
        with h5py.File(h5) as h5:
            molids = h5['molids'][:]
            failed_molids = h5['failed_molids'][:]
            i2s = h5['i2s'][:]  # TODO: this should be lazy / attached to the HDF5 storage
            data = h5['data'][:]
            indices = h5['indices'][:]
            indptr = h5['indptr'][:]
            shape = h5['shape'][:]
            csr = csr_matrix((data, indices, indptr), shape=shape)
            return UnfoldedFingerprints(molids, i2s, csr, failed_molids=failed_molids)

    def X01(self, inplace=False):
        """Returns X, but the data is now binary."""
        if inplace:
            self.csr.data = np.ones(len(self.csr.data), dtype=np.bool)
            return self.csr
        return csr_matrix((np.ones(len(self.csr.data), dtype=np.bool), self.csr.indices, self.csr.indptr),
                          shape=self.csr.shape)

    def substructures(self, f_numbers):
        if not isinstance(f_numbers, collections.Iterable):
            f_numbers = [f_numbers]
        return self.i2s[f_numbers]


def find_sparse_dupes(matrix, by_rows=True):
    """
    Find exact duplicates within the rows or columns of an scipy sparse matrix.
    (simple attempt to reduce correlation in the dataset by zeroing duplicate features)

    Parameters
    ----------
    matrix: scipy sparse matrix
        The matrix, it will be transformed to csr or csc as needed.

    by_rows: boolean, default True
        If True, duplicates among rows are returned; if False, work on columns.

    Returns
    -------
    A list of *python* arrays, each containing the indices of a group of duplicates.
    Each duplicates group is sorted from lower index to higher, and so it is sorted the top level list.

    Examples
    --------

    >>> matrix = coo_matrix(([1] * 6, ((1, 1, 2, 3, 4, 4), (2, 3, 1, 2, 2, 3))))
    >>> groups = find_sparse_dupes(matrix)
    >>> len(groups), len(groups[0]), len(groups[1]), len(groups[2]), len(groups[3])
    (4, 1, 2, 1, 1)
    >>> groups[0][0], groups[1][0], groups[1][1], groups[2][0], groups[3][0]
    (0, 1, 4, 2, 3)
    """
    if by_rows:
        matrix = matrix.tocsr()
    else:
        matrix = matrix.tocsc()
    dupes = defaultdict(lambda: defaultdict(lambda: array('i')))
    num_rows, num_cols = matrix.shape
    for i in xrange(num_rows if by_rows else num_cols):
        data = matrix.indices[matrix.indptr[i]:matrix.indptr[i+1]]
        sha512 = hashlib.sha512(data.data).hexdigest()
        sha256 = hashlib.sha256(data.data).hexdigest()
        dupes[sha512][sha256].append(i)
    return sorted(chain(*[d.values() for d in dupes.values()]))


def zero_dupes(matrix, by_rows=True, copy=True):
    """Zero all but one of the representants of each group in the matrix.

    Returns
    -------
    A csr matrix with all the duplicates rows / cols zeroed.

    Examples
    --------

    >>> matrix = coo_matrix(([1] * 6, ((1, 1, 2, 3, 4, 4), (2, 3, 1, 2, 2, 3))))
    >>> matrix = zero_dupes(matrix)
    >>> len(matrix.data)
    4
    >>> matrix[1, 2], matrix[1, 3], matrix[2, 1], matrix[3, 2], matrix[4, 2], matrix[4, 3]
    (1, 1, 1, 1, 0, 0)
    """
    dupes = find_sparse_dupes(matrix, by_rows=by_rows)
    # Rewrite the matrix
    if copy:
        matrix = matrix.copy()  # Inefficient
    # if by_rows:
    #     matrix = matrix.tocsr()
    # else:
    #     matrix = matrix.tocsc()
    # TODO: optimize recreating indices / indptr / data from selected numpy slices
    matrix = matrix.tolil()
    for group in dupes:
        for to_zero in group[1:]:
            if by_rows:
                matrix[to_zero, :] = 0
            else:
                matrix[:, to_zero] = 0
    # matrix.eliminate_zeros()
    return matrix.tocsr()


@whatable
class RDKMorganFingerprinter(object):

    def __init__(self,
                 fcfp=False,
                 max_radius=30,
                 use_hs=False,
                 canonical=True,
                 isomeric=False,
                 kekule=False,
                 all_bonds_explicit=False):
        super(RDKMorganFingerprinter, self).__init__()

        # Configuration
        self.fcfp = fcfp
        self.max_radius = max_radius
        self.use_hs = use_hs
        self.canonical = canonical
        self.isomeric = isomeric
        self.kekule = kekule
        self.all_bonds_explicit = all_bonds_explicit

        # Data
        self._molids = []
        self._failed_moldids = []
        self._s2i = {}
        self._rows = array('I')
        self._cols = array('I')
        self._vals = array('I')

    def add_mol(self, molid, rdkmol):
        """Serial, memory-conservative fingerprinter for building a fingerprints sparse matrix incrementally."""

        try:
            # Amazing DRY violation
            counts, _ = unfolded_morgan_fingerprint(rdkmol,
                                                    max_radius=self.max_radius,
                                                    fcfp=self.fcfp,
                                                    use_hs=self.use_hs,
                                                    canonical=self.canonical,
                                                    isomeric=self.isomeric,
                                                    kekule=self.kekule,
                                                    all_bonds_explicit=self.all_bonds_explicit)

            cols_vals = []
            for smiles, count in counts.iteritems():
                if smiles not in self._s2i:
                    self._s2i[smiles] = len(self._s2i)
                self._rows.append(len(self._molids))
                cols_vals.append((self._s2i[smiles], count))

            cols_vals = sorted(cols_vals)
            self._cols.extend(map(itemgetter(0), cols_vals))
            self._vals.extend(map(itemgetter(1), cols_vals))
            self._molids.append(molid)
        except Exception, _:
            warning('Could not compute unfolded fingerprint for molecule %s' % molid)
            self._failed_moldids.append(molid)

    def fingerprints(self):
        i2s = [smiles for smiles, _ in sorted(self._s2i.items(), key=itemgetter(1))]
        csr = coo_matrix((self._vals, (self._rows, self._cols)),
                         shape=(len(self._molids), len(self._s2i))).tocsr()
        return UnfoldedFingerprints(self._molids, i2s, csr, failed_molids=self._failed_moldids)

if __name__ == '__main__':
    from manysources.datasets import ManysourcesDataset, MANYSOURCES_MOLECULES
    for dset in MANYSOURCES_MOLECULES.keys():
        print dset
        dset = ManysourcesDataset(dset)
        ufp = dset.ecfps()
        molids, X, Y = dset.ecfps_molidsXY()
        # This should find "duplicate" features
        dupe_columns = find_sparse_dupes(X, by_rows=False)
        for group in dupe_columns:
            if len(group) > 1:
                print 'Duplicated features: %s' % ' '.join(map(str, group))
                print '\t%s' % ' '.join(ufp.substructures(list(group)))
        # This should remove "duplicate" features
        # nnz_before = X.nnz
        # X = zero_dupes(X, by_rows=False)
        # print 'Before: %d; After: %d' % (nnz_before, X.nnz)
        # This should find duplicate rows
        dupe_rows = find_sparse_dupes(X, by_rows=True)  # Groups of duplicates...
        for group in dupe_rows:
            if len(group) > 1:
                print 'Dupes: %s' % ' '.join(molids[list(group)])
        print '-' * 80
        print '#' * 80