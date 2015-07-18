# coding=utf-8
"""Molecule manipulation using rdkit."""
from __future__ import print_function
from collections import Iterable
from collections import defaultdict

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.Chem.PropertyMol import PropertyMol
from whatami import whatable

from manysources import warning
from manysources.common.sdf import iterate_records_in_text_file


# --- DESCRIPTOR COMPUTATION
# For a list of available descriptors see:
#  - http://www.rdkit.org/docs/GettingStartedInPython.html#list-of-available-descriptors
#  - http://code.google.com/p/rdkit/wiki/DescriptorsInTheRDKit
#  - http://www.rdkit.org/docs/api/rdkit.Chem.Descriptors-module.html


def discover_rdk_descriptors(verbose=False):
    """
    Returns a list of the names descriptors present in RDKIT.
    (other than fps, ultrashape, partial charges, topliss and possibly others)
    """
    # TODO: many of these descriptors have parameters
    descriptors = tuple([desc_name for desc_name, func in Descriptors._descList])
    if verbose:
        print('Discovered RDKIT descriptors...')
        print('\n'.join(descriptors))
        print('-' * 80)
        print('Members of class Descriptors that are not descriptors...')
        print('\n'.join(sorted(set(Descriptors.__dict__.keys()) -
                               set(descriptors))))
    return descriptors


def desc_wrapper(desc_func):  # TODO: allow logging
    def wrapper(mol):
        try:
            return desc_func(mol)
        except:
            return np.nan
    return wrapper


@whatable
class RDKitDescripter(object):

    def __init__(self, descriptors=None):
        super(RDKitDescripter, self).__init__()
        self.descriptors = discover_rdk_descriptors() if descriptors is None else descriptors
        self._dfs = [desc_wrapper(getattr(Descriptors, descriptor)) for descriptor in self.descriptors]

    def _compute_for_mol(self, mol):
        return np.array([computer(mol) for computer in self._dfs])

    def compute(self, mols):
        if not isinstance(mols, Iterable):
            mols = [mols]
        X = np.empty((len(mols), len(self._dfs)))
        for i, mol in enumerate(mols):
            X[i, :] = self._compute_for_mol(mol)
        return X

    def fnames(self, prefix='rdkit-'):
        return ['%s%s' % (prefix, d) for d in self.descriptors]


##################################################
# FINGERPRINT COMPUTATION
# For a list of available fingerprints see:
#  - https://code.google.com/p/rdkit/wiki/FingerprintsInTheRDKit
# Add also the avalon fingerprints exposed via the rdkit bindings
#  - http://www.rdkit.org/docs/api/rdkit.Avalon.pyAvalonTools-module.html
##################################################

##################################################
# MORGAN FINGERPRINTS
# See:   http://www.rdkit.org/docs/GettingStartedInPython.html#morgan-fingerprints-circular-fingerprints
##################################################

def explain_circular_substructure(mol,
                                  center,
                                  radius,
                                  use_hs=False,
                                  canonical=True, isomeric=False, kekule=False, all_bonds_explicit=False):
    """Returns a SMILES description of the circular structure defined by a center and a topological radius."""
    atoms = {center}
    env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, center, useHs=use_hs)
    for bidx in env:
        bond = mol.GetBondWithIdx(bidx)
        atoms.add(bond.GetBeginAtomIdx())
        atoms.add(bond.GetEndAtomIdx())
    return Chem.MolFragmentToSmiles(mol,
                                    atomsToUse=list(atoms),
                                    bondsToUse=env,
                                    rootedAtAtom=center,
                                    isomericSmiles=isomeric,
                                    kekuleSmiles=kekule,
                                    canonical=canonical,
                                    allBondsExplicit=all_bonds_explicit)


def unfolded_morgan_fingerprint(mol,
                                max_radius=100,
                                fcfp=False,
                                use_hs=False,
                                canonical=True,
                                isomeric=False,
                                kekule=False,
                                all_bonds_explicit=False):
    if isinstance(mol, basestring):
        mol = to_rdkit_mol(mol)
    fpsinfo = {}
    # N.B. We won't actually use rdkit hash, so we won't ask for nonzero values...
    # Is there a way of asking rdkit to give us this directly?
    AllChem.GetMorganFingerprint(mol, max_radius, bitInfo=fpsinfo, useFeatures=fcfp)
    counts = defaultdict(int)
    centers = defaultdict(list)
    for bit_descs in fpsinfo.values():
        for center, radius in bit_descs:
            cansmiles = explain_circular_substructure(mol, center, radius,
                                                      use_hs=use_hs,
                                                      canonical=canonical,
                                                      isomeric=isomeric,
                                                      kekule=kekule,
                                                      all_bonds_explicit=all_bonds_explicit)
            counts[cansmiles] += 1
            centers[cansmiles].append((center, radius))
    return counts, centers


# --- Instantiation of molecules

def to_rdkit_mol(smiles, molid=None, sanitize=True, to2D=False, to3D=False, toPropertyMol=False):
    """Converts a smiles string into an RDKit molecule."""
    mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)  # TODO: allow other formats, like sdf or inchi
    if mol is None:
        if molid is None:
            warning('RDKit cannot create a molecule from smiles %s' % smiles)
        else:
            warning('RDKit cannot create molecule %s from smiles %s' % (molid, smiles))
        return mol
    if to3D:
        AllChem.EmbedMolecule(mol)
        AllChem.UFFOptimizeMolecule(mol)
    elif to2D:
        AllChem.Compute2DCoords(mol)
    if toPropertyMol:
        return PropertyMol(mol)
    return mol


def molecule_iterator(filename):
    return Chem.SupplierFromFilename(filename)


def iterate_rdkit_mols_in_sdf_file(filename,
                                   sanitize=True,
                                   record_terminator='$$$$',
                                   terminator_only_after_empty_line=True,
                                   verbose=True):
    from rdkit.Chem import SDMolSupplier
    # MolFromMolBlock won't do...
    # http://www.mail-archive.com/rdkit-discuss@lists.sourceforge.net/msg01436.html
    supplier = SDMolSupplier()

    def mol_from_molblock(molblock):
        supplier.SetData(molblock, sanitize=sanitize)
        return supplier.next()

    return ((mol_from_molblock(molblock), molblock) for molblock in
            iterate_records_in_text_file(filename,
                                         record_terminator=record_terminator,
                                         terminator_only_after_empty_line=terminator_only_after_empty_line,
                                         verbose=verbose))


##################################################
# Topological fingerprints
# See: http://www.rdkit.org/docs/GettingStartedInPython.html#topological-fingerprints
#      http://www.rdkit.org/docs/api/rdkit.Chem.rdmolops-module.html#RDKFingerprint
# This line of code shows how to get a SMILES string for each substructure easily
#  https://github.com/rdkit/rdkit/blob/master/Code/GraphMol/Fingerprints/Fingerprints.cpp#L532
# Note that each subgraph can set more than one bit (governed by a parameter),
# effectively reducing the evil effect of collisions.
##################################################
#
# RDKFingerprint( ... ) -> ExplicitBitVect :
#     Returns an RDKit topological fingerprint for a molecule
#
#       Explanation of the algorithm below.
#
#       ARGUMENTS:
#
#         - mol: the molecule to use
#
#         - minPath: (optional) minimum number of bonds to include in the subgraphs
#           Defaults to 1.
#
#         - maxPath: (optional) maximum number of bonds to include in the subgraphs
#           Defaults to 7.
#
#         - fpSize: (optional) number of bits in the fingerprint
#           Defaults to 2048.
#
#         - nBitsPerHash: (optional) number of bits to set per path
#           Defaults to 2.
#
#         - useHs: (optional) include paths involving Hs in the fingerprint if the molecule
#           has explicit Hs.
#           Defaults to True.
#
#         - tgtDensity: (optional) fold the fingerprint until this minimum density has
#           been reached
#           Defaults to 0.
#
#         - minSize: (optional) the minimum size the fingerprint will be folded to when
#           trying to reach tgtDensity
#           Defaults to 128.
#
#         - branchedPaths: (optional) if set both branched and unbranched paths will be
#           used in the fingerprint.
#           Defaults to True.
#
#         - useBondOrder: (optional) if set both bond orders will be used in the path hashes
#           Defaults to True.
#
#         - atomInvariants: (optional) a sequence of atom invariants to use in the path hashes
#           Defaults to empty.
#
#         - fromAtoms: (optional) a sequence of atom indices. If provided, only paths/subgraphs
#           starting from these atoms will be used.
#           Defaults to empty.
#
#         - atomBits: (optional) an empty list. If provided, the result will contain a list
#           containing the bits each atom sets.
#           Defaults to empty.
#
#       RETURNS: a DataStructs.ExplicitBitVect with _fpSize_ bits
#
#       ALGORITHM:
#
#        This algorithm functions by find all subgraphs between minPath and maxPath in
#         length.  For each subgraph:
#
#          1) A hash is calculated.
#
#          2) The hash is used to seed a random-number generator
#
#          3) _nBitsPerHash_ random numbers are generated and used to set the corresponding
#             bits in the fingerprint
#
##################################################
