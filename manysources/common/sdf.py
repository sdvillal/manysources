# coding=utf-8
"""Utilities to work with compounds collections in SDF format."""
import gzip
import zipfile
import os.path as op
import os
import glob

from manysources import warning
from manysources.common.misc import ensure_dir


def iterate_records_in_text_file(filename,
                                 record_terminator='$$$$',
                                 terminator_only_after_empty_line=True,
                                 verbose=True):
    """Iterate over all molecule records in the given sdf file.
    # NB: assume that the last record also includes terminator
    # Must the record terminator be preceded by an empty line?
    """
    #Let's detect compressed files just using the extension
    if hasattr(filename, 'next'):
        fileh = filename
    elif filename.endswith('.zip'):
        fileh = zipfile.ZipFile(filename, 'r')
    elif filename.endswith('.gz'):
        fileh = gzip.GzipFile(filename, 'r')
    else:
        fileh = open(filename, 'r')
    notatend = True
    partial_record = []
    line_num = 0
    while notatend:
        notatend = fileh.next()
        partial_record.append(notatend)
        line_num += 1
        if notatend.strip() == record_terminator:
            if terminator_only_after_empty_line:
                if len(partial_record) < 2 or partial_record[-2] != '\n':
                    if verbose:
                        warning('Warning: the terminator %s in line %d is not preceded by a blank line\n' %
                                (partial_record[-1].strip(), line_num))
                    continue  # Do not yield yet, as the terminator is not preceded by an empty line
            full_record = partial_record
            partial_record = []
            yield ''.join(full_record).strip()
    fileh.close()


def split_sdf(input_sdf, output_dir, prefix, nb_mols=10000):

    mol_iterator = iterate_records_in_text_file(input_sdf)
    ensure_dir(output_dir)

    def write_n_mols(fn):
        mols_counter = 0
        with open(fn, 'w') as writer:
            for mol in mol_iterator:
                writer.write(mol)
                writer.write('\n')
                mols_counter += 1
                if mols_counter == nb_mols:
                    return True
        return False

    def fn(fnum, pad=None):
        if not pad:
            return op.join(output_dir, '%s_%d.sdf' % (prefix, fnum))
        else:
            return op.join(output_dir, prefix + '_' + pad + str(fnum) + '.sdf')

    file_counter = 1
    while True:
        if not write_n_mols(fn(file_counter)):
            break
        file_counter += 1

    #check if last file is empty (not elegant but easy for me)
    last_file = op.join(output_dir, '%s_%d.sdf' % (prefix, file_counter))
    if op.getsize(last_file) == 0:
        os.remove(last_file)
        file_counter -= 1

    #rename the files with 0 padding so that we keep numerical order
    padding = len(str(file_counter))
    files_to_rename = glob.glob(op.join(output_dir, '*.sdf'))
    for f in files_to_rename:
        file_nb = f.split('_')[-1]
        file_nb = file_nb.split('.')[0]
        pad = '0'*(padding-len(str(file_nb)))
        if len(pad) > 0:
            os.rename(f, fn(file_nb, pad=pad))


def count_mols_in_sdf(sdf):
    """Returns the number of molecules in an sdf file as the number of '$$$$'.
    Do not pass a big file!
    TODO: use grep...
    """
    with open(sdf) as reader:
        return reader.read().count('$$$$')