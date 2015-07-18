"""
This source is very simple: if there is a -OH, the compounds are inactive. Else, they are active. So the substructures
encoding this -OH should have a big influence in properly predicting the molecules in this specific source. If I check
the picture, these are the substructures that most change weight and could encode this hydroxy substitution:

    c(c(c)O)c(c)O
    c(O)(cc)c(c)c
    c(O)(cc)cc

What are their weights when the source is in? Is it positive? Is it negative?
What are their weights when the source is out?
How well/badly are predicted molecules 2d, 1d, 2c, 2a and 2b? (the inactives)
How well/badly are predicted molecules 1e, 1c, 1a, 1d, 4d, 4b, 1b, 4c, 4a? (the inhibitors)
Is there a relationship between the change of weight of those specific substructures in a given expid and the way the molecules are predicted?


"""
from datasets import ManysourcesDataset
from experiments import ManysourcesResult
from manysources import MANYSOURCES_ROOT
import numpy as np
import pandas as pd
import os.path as op
import h5py
import matplotlib.pyplot as plt


source = 'acridones_Boumendjel_2007'
substructures = {'acridones_Boumendjel_2007': ['c(c(c)O)c(c)O', 'c(O)(cc)c(c)c', 'c(O)(cc)cc', 'C(Cc)NC','c1(OC)cccc(c)c1'],
                 'Cramer_2007': ['C(CN)N(C)c', 'N(C)(C)C', 'C(C)(C)(c)O', 'N(C)(C)c', 'C(CN)N(C)C', 'C(=O)(c(c)c)N(C)C',
                                 'c(cc)(cc)C(N)=O', 'c(cc)(cc)NC', 'c(cc)(cc)[N+](=O)[O-]', 'C(CC)CC', 'c(cc)(cc)C(c)=O',
                                 'C(C)N(C)C', 'C(c)(N)=O', 'c(c)(c)F', 'c1ccccc1C(N)=O', 'C(CC)N(C)C', 'C1CCCN(C)C1',
                                 'O(C)C(c)=O', 'C(c)(F)(F)F'],
                 'Sugimoto_2003':['C(C)S', 'c1cc(OCCN)ccc1C(=C(CC)c(cc)cc)c1ccccc1', 'C(Oc)c(c)c','N(C)(C)C','C(C)C(=C)c'],
                 'Boumendjel_2005': ['C(C)(C)C', 'C(C)(C)(C)O', 'C(CC)CN', 'C(Cc(c)c)NC(c)=O','c1(OC)ccccc1OC',
                                     'N(C)(C)C', 'C(CNC)c(cc)cc'],
                 'Ochoa-Puentes_2011': ['N(C(=O)c(c)c)c(cc)cc', 'C(=O)(Nc)c(c)n', 'C(O)(=O)c(c)c', 'N(C)(C)C',
                                        'c(CN)(cc)c(C)c'],
                 'Pick_2008': ['C(C)(NC)C(N)=O', 'C(=O)(NC)C(C)N', 'c1cccc(C)c1[N+](=O)[O-]'],
                 'Ahmed-Belkacem_2007': ['C(C)(C)(c)O', 'c1(O)cc(O)cc(o)c1c(c)=O'],
                 'Patel_2011': ['c1cccc(C)c1NC', 'c(cc)(cc)[N+](=O)[O-]'],
                 'Pick_2010': ['N(C(=O)c(cc)cc)c1ccccc1C(N)=O', 'c(cc)(cc)[N+](=O)[O-]',
                               'c1cc([N+](=O)[O-])ccc1C(=O)Nc(c)c', 'c1(NC(c)=O)ccccc1C(N)=O'],
                 'Pick_2011': ['C1(O)C(C)OCC(O)C1O', 'C(C)(C)(O)O'],
                 'Takada_2010': ['C(C(C)O)c(c)c', 'Brc', 'C(Cc)NC', 'C(=Cc1ccc(O)cc1)NC(=O)C(=Cc(cc)cc)OC'],
                 'flavonoids_Zhang_2004': ['C1(O)C(C)OCC(O)C1O', 'C(=O)(OC)c(c)c'],
                 'Jin_2006': ['C1(O)C(O)C(C)OC(O)C1O', 'C(OC)(OC)C(C)O'],
                 'Bioorg. Med. Chem. Lett., (2009) 19:17:5078': ['C1CC1C', 'C(F)(F)(F)c(c)c', 'C(C(C)C)N(C(C)C)C(c)=O',
                                                                 'c(cc)(C(N)=O)c(c)Cl'],
                 'Bioorg. Med. Chem. Lett., (2011) 21:12:3627':['C1CCN(C)C1CN', 'C1CN(C)CCO1', 'N(CC)c(c)c'],
                 'Bioorg. Med. Chem. Lett., (2006) 16:20:5303':['N1(C(=O)N(C)C)CCNCC1', 'C1CCN(C)C1CN'],
                 'Bioorg. Med. Chem. Lett., (2008) 18:19:5280':['N1(C)CCCCC1', 'C(CC)Oc', 'N#C', 'C(CC)(CC)Oc'],
                 'Bioorg. Med. Chem., (2011) 19:2:883': ['c(cc)(cc)C(F)(F)F', 'n1(-c(cc)cc)ccccc1=O', 'n(c)(c)c'],
                 'ACS Med. Chem. Lett., (2011) 2:12:913': ['C(NC)C(N)=O', 'c(cc)(cc)C(F)(F)F', 'c(-c)(n)n',
                                                           'N(CC)(CC)c(c)n', 'C(=O)(CN)N(C)C'],
                 'Bioorg. Med. Chem. Lett., (2012) 22:6:2242':['c1cc(C)ccc1F', 'C(CC)(NC)c(c)c', 'N(C)O'],
                 'Jimenez-Alonso_2008': ['Brc','c1cc(C)ccc1OC','C(C)(C)(C)O', 'C(C)(C)=C','c1c(OC)ccc(C)c1OC'],
                 'cdkinhib_An_2008': ['c1c(nc)c(c)cc(O)c1OC','n(c(c)N)c(N)n', 'c(CN)(cc)cc'],
                 'Bokesch_2010': ['c(oc)(c(c)c)c(c)c','c(O)(c(c)c)c(c)c','C(C)C','c(OC)(c(c)c)c(c)O']}
no_impact = {'acridones_Boumendjel_2007': 'C(c)(F)(F)F', 'Cramer_2007':'C(=O)(CC)c(c)c', 'Boumendjel_2005':'C(c)(N)=O'}
PROBLEMATIC_EXPIDS = {'bcrp':[489, 1840, 2705, 2780, 3842]}

def to_short_name_bcrp(long_name):
    return long_name.split('n=')[1]


def to_short_name_herg(long_name):
    return long_name.split('CHEMBL')[1]


def to_short_name_pgp_cruciani(long_name):
    return abs(hash(long_name)) % (10 ** 8)


SHORT_NAMES = {'bcrp': to_short_name_bcrp, 'hERG': to_short_name_herg, 'pgp-cruciani': to_short_name_pgp_cruciani}


def get_weights_before_after(substruct, source, dset='bcrp', model='logreg3', feats='ecfps1', num_expids=4096):
    from manysources.analyses.substructures import substructs_weights_one_source
    s2i = ManysourcesDataset(dset).ecfps(no_dupes=True).s2i
    index_substruct = s2i[substruct]
    expids, all_in, all_out = substructs_weights_one_source(source, model=model, feats=feats, dset=dset,
                                                            num_expids=num_expids)
    return expids, all_in[:,index_substruct], all_out[:,index_substruct]


def violin_plot_weights_in_out(substruct, source, dset='bcrp', model='logreg3', feats='ecfps1', num_expids=4096):
    expids, weights_in, weights_out = get_weights_before_after(substruct, source, dset=dset, model=model, feats=feats,
                                                       num_expids=num_expids)
    print weights_in.shape, weights_out.shape

    df = pd.DataFrame.from_dict({'weights_in': weights_in, 'weights_out': weights_out})
    df_tidy = pd.melt(df, value_name='weight')
    print df_tidy
    import seaborn
    seaborn.set_style("ticks")
    seaborn.set_context("talk")
    seaborn.set_palette('deep')
    seaborn.violinplot('variable', 'weight', data=df_tidy)
    plt.title('Weights for substructure %s'%substruct)
    plt.show()


def read_losses_one_source(source, dset, model, feats, lso=True, expids=tuple(range(4096))):
    molids_in_source = ManysourcesDataset(dset).mols().source2molids(source)

    cache_path = op.join(MANYSOURCES_ROOT, 'data', 'results', 'square_losses.h5')
    result_coords = '/dset=%s/feats=%s/model=%s/lso=%r/score_calibration=\'0-1\'' % (dset, feats, model, lso)
    with h5py.File(cache_path, 'r') as h5:
        group = h5[result_coords]
        infile_expids = group['expids'][()] if expids is not None else expids
        if 0 == len(set(expids) - set(infile_expids[:, 0])):
            e2r = {e: i for e, i in infile_expids if i >= 0}
            ok_expids = [expid for expid in expids if expid in e2r]
            rows = [e2r[expid] for expid in ok_expids]
            losses = group['losses'][rows].T
            molids = group['molids'][:]
            indices_of_molids = [np.where(molids==molid)[0][0] for molid in molids_in_source]
            losses = losses[indices_of_molids, :].T
            if 'n=' in molids[0]:
                molids = map(lambda x: x.split('n=')[1], molids[indices_of_molids])
            else:
                molids = molids[indices_of_molids]
            return pd.DataFrame(losses, columns=molids, index=ok_expids)


def violin_plot_losses_source(source, dset='bcrp', model='logreg3', feats='ecfps1', expids=tuple(range(4096))):

    df_lso = read_losses_one_source(source=source, dset=dset, model=model, feats=feats, expids=expids)
    df_lso['expid'] = df_lso.index
    df_lso['lso'] = [True for _ in range(len(df_lso))]
    df_lso = pd.melt(df_lso, id_vars=['expid', 'lso'], var_name='molid')
    df_crs = read_losses_one_source(source, dset=dset, model=model, feats=feats, expids=expids, lso=False)
    df_crs['expid'] = df_crs.index
    df_crs['lso'] = [False for _ in range(len(df_crs))]
    df_crs = pd.melt(df_crs, id_vars=['expid', 'lso'], var_name='molid')
    merged_tidy_df = pd.concat([df_lso, df_crs])

    print merged_tidy_df

    colors = ['green', 'red', 'red', 'red', 'green', 'green', 'red', 'red', 'red', 'red', 'red', 'green', 'green']
    import seaborn
    seaborn.set_style("ticks")
    seaborn.set_context("talk")
    #seaborn.set_palette(seaborn.color_palette(colors))
    #seaborn.palplot(seaborn.color_palette(colors))
    seaborn.set_palette('deep')
    seaborn.violinplot('molid', 'value', 'lso', data=merged_tidy_df, split=True, scale='width')
    plt.title('Losses across all expids for the compounds in acridones_Boumendjel_2007,\n with LSO split or Random split')
    plt.ylim([0,1])
    # from matplotlib import patches
    # red_patch = patches.Patch(color='red', label='inhibitor')
    # green_patch = patches.Patch(color='green', label='inactive')
    # plt.legend(handles=[red_patch, green_patch])
    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()


def violin_two_plots_losses_source(source, dset='bcrp', model='logreg3', feats='ecfps1', expids=tuple(range(4096)),
                                   inactives=None, inhibitors=None, short_names=True):
    if inactives is None:
        inactives = ['2d', '1d', '2c', '2a', '2b']
    if inhibitors is None:
        inhibitors = ['1e', '1c', '1a', '4d', '4b', '1b', '4c', '4a']
    df_lso = read_losses_one_source(source=source, dset=dset, model=model, feats=feats, expids=expids)
    df_lso['expid'] = df_lso.index
    df_lso['lso'] = [True for _ in range(len(df_lso))]
    df_lso = pd.melt(df_lso, id_vars=['expid', 'lso'], var_name='molid')
    df_crs = read_losses_one_source(source, dset=dset, model=model, feats=feats, expids=expids, lso=False)
    df_crs['expid'] = df_crs.index
    df_crs['lso'] = [False for _ in range(len(df_crs))]
    df_crs = pd.melt(df_crs, id_vars=['expid', 'lso'], var_name='molid')
    merged_tidy_df = pd.concat([df_lso, df_crs])
    if short_names:
        merged_tidy_df['molid'] = map(SHORT_NAMES[dset], merged_tidy_df['molid'])
    tidy_df_inhibitors = pd.concat([merged_tidy_df.loc[merged_tidy_df.molid == inhibitor] for inhibitor in inhibitors])
    tidy_df_inactives = pd.concat([merged_tidy_df.loc[merged_tidy_df.molid == inactive] for inactive in inactives])

    import seaborn
    seaborn.set_style("ticks")
    seaborn.set_context("talk")
    fig, (ax1, ax2) = plt.subplots(ncols=2)
    seaborn.violinplot('molid', 'value', 'lso', data=tidy_df_inactives, split=True, scale='width', ax=ax1, palette='Greens')
    seaborn.violinplot('molid', 'value', 'lso', data=tidy_df_inhibitors, split=True, scale='width', ax=ax2, palette='Reds')
    #locs, labels = plt.xticks()
    #plt.setp(labels, rotation=90)
    ax1.set_ylim([0,1])
    ax2.set_ylim([0,1])
    plt.show()


def violin_two_plots_losses_source_one_class(source, dset='bcrp', model='logreg3', feats='ecfps1', expids=tuple(range(4096)),
                                   inactives=None, inhibitors=None, short_names=True):
    if inactives is None:
        palette ='Reds'
    elif inhibitors is None:
        palette = 'Greens'
    df_lso = read_losses_one_source(source=source, dset=dset, model=model, feats=feats, expids=expids)
    df_lso['expid'] = df_lso.index
    df_lso['lso'] = [True for _ in range(len(df_lso))]
    df_lso = pd.melt(df_lso, id_vars=['expid', 'lso'], var_name='molid')
    df_crs = read_losses_one_source(source, dset=dset, model=model, feats=feats, expids=expids, lso=False)
    df_crs['expid'] = df_crs.index
    df_crs['lso'] = [False for _ in range(len(df_crs))]
    df_crs = pd.melt(df_crs, id_vars=['expid', 'lso'], var_name='molid')
    merged_tidy_df = pd.concat([df_lso, df_crs])
    if short_names:
        merged_tidy_df['molid'] = map(SHORT_NAMES[dset], merged_tidy_df['molid'])
    if inhibitors is not None:
        merged_tidy_df = pd.concat([merged_tidy_df.loc[merged_tidy_df.molid == inhibitor] for inhibitor in inhibitors])
    elif inactives is not None:
        merged_tidy_df = pd.concat([merged_tidy_df.loc[merged_tidy_df.molid == inhibitor] for inhibitor in inactives])

    import seaborn
    seaborn.set_style("ticks")
    seaborn.set_context("talk")
    seaborn.violinplot('molid', 'value', 'lso', data=merged_tidy_df, split=True, scale='width', palette=palette)
    plt.ylim([0,1])
    plt.show()


def violin_plot_manager(source, dset, model, feats, expids, inhibitors, inactives, short_names=True):
    if len(inactives) == 0:
        print 'This is a monoactivity source...'
        violin_two_plots_losses_source_one_class(source, dset=dset, model=model, feats=feats, expids=expids,
                                                 inhibitors=inhibitors, short_names=short_names)
    elif len(inhibitors) == 0:
        print 'This is a monoactivity source...'
        violin_two_plots_losses_source_one_class(source, dset=dset, model=model, feats=feats, expids=expids,
                                                 inactives=inactives, short_names=short_names)
    else:
        print 'Making a 2-classes violin plot...'
        violin_two_plots_losses_source(source, dset=dset, model=model, feats=feats, expids=expids, inactives=inactives,
                                       inhibitors=inhibitors, short_names=short_names)


def automatic_violin_two_plots(dset='bcrp', model='logreg3', feats='ecfps1', expids=tuple(range(4096)),
                               short_names=True):
    sources = list(ManysourcesDataset(dset=dset).molecules().present_sources())
    for source in sources:
        print source
        inact, inhib = map_short_name_to_activity(source, dset=dset, short_names=short_names)
        print inact, inhib
        if inact is None and inhib is None:
            print 'Source with only 1 molecule, skipping...'
            continue
        #elif len(inact) + len(inhib) > 5:  # <=
        #    print 'Probably already plotted, skipping...'
        #    continue

        # Is the source big?
        if len(inact) + len(inhib) > 30:
            print 'This is a large source...'
            # 1. Cluster molecules
            from rdkit.Chem import AllChem
            from manysources.sandbox.cluster_mols import cluster_fps, post_process_clusters
            molids_in_source = ManysourcesDataset(dset=dset).molecules().source2molids(source)
            mols_in_source = [ManysourcesDataset(dset=dset).molecules().molid2mol(molid) for molid in molids_in_source]
            print len(mols_in_source)
            fps = [AllChem.GetMorganFingerprintAsBitVect(mol,4,1024) for mol in mols_in_source]
            clusters = cluster_fps(fps,cutoff=0.5)
            print clusters
            new_clusts = post_process_clusters(clusters, groups=[])
            # 2. Create a separate violin plot per cluster
            for group in new_clusts:
                print group
                group = np.array(group)
                group_molids = np.array(molids_in_source)[group]
                if short_names:
                    inactives = [molid for molid in inact if molid in map(SHORT_NAMES[dset], group_molids)]
                    inhibitors = [molid for molid in inhib if molid in map(SHORT_NAMES[dset], group_molids)]
                else:
                    inactives = [molid for molid in inact if molid in group_molids]
                    inhibitors = [molid for molid in inhib if molid in group_molids]
                print inactives, inhibitors
                violin_plot_manager(source, dset=dset, model=model, feats=feats, expids=expids,
                                               inactives=inactives, inhibitors=inhibitors, short_names=short_names)
        else:
            violin_plot_manager(source, dset=dset, model=model, feats=feats, expids=expids, inactives=inact,
                                inhibitors=inhib, short_names=short_names)


# Very anecdotical, but does the loss goes down for the inactives in the few expids where c(O)(cc)c(c)c is not null?

def coevolution_loss_weight(molid, substruct, source, dset='bcrp', model='logreg3', feats='ecfps1'):
    df_lso = read_losses_one_source(source=source, dset=dset, model=model, feats=feats)
    losses = list(df_lso[molid])[:4096]
    expids, _, weights_out = get_weights_before_after(substruct, source=source, dset=dset, model=model, feats=feats, num_expids=4096)
    df_bla = pd.DataFrame.from_dict({'losses': losses, 'weights_out': weights_out})
    df_bla['expid'] = expids
    df_bla_tidy = pd.melt(df_bla, id_vars=['expid'])
    print df_bla_tidy
    import seaborn
    seaborn.set_style("ticks")
    seaborn.set_context("talk")
    seaborn.set_palette('deep')
    #seaborn.factorplot('expid', y='value', data=df_bla_tidy, hue='variable')
    seaborn.lmplot('weights_out', 'losses', data=df_bla)
    plt.show()


def map_short_name_to_activity(source, dset='bcrp', short_names=True):
    molids_in_source = np.array(ManysourcesDataset(dset).mols().source2molids(source))
    molids2class = np.array(map(ManysourcesDataset(dset).mols().molid2label, molids_in_source))
    if len(molids_in_source) < 2:
        return None, None
    if short_names:
        s_names = np.array(map(SHORT_NAMES[dset], molids_in_source))
        #return short_names[molids2class == 'INACTIVE'], short_names[molids2class == 'INHIBITOR']
        return s_names[molids2class == '0'], s_names[molids2class == '1']
    else:
        return molids_in_source[molids2class == '0'], molids_in_source[molids2class == '1']


if __name__ == '__main__':
    #violin_plot_weights_in_out('C(Cc1ccc(NC(=O)c(c)c)cc1)N1CCc2cc(OC)c(OC)cc2C1', 'Ochoa-Puentes_2011')
    #violin_plot_losses_source(source, dset='bcrp', model='logreg3', feats='ecfps1')
    #coevolution_loss_weight('1c', 'c(O)(cc)c(c)c', source)
    #inact, inhib = map_short_name_to_activity('flavonoids_Zhang_2004', dset='bcrp', short_names=True)
    #violin_two_plots_losses_source_one_class('flavonoids_Zhang_2004', dset='bcrp', inactives=inact)
    #violin_two_plots_losses_source('ACS Med. Chem. Lett., (2011) 2:12:913', dset='hERG', model='logreg3', feats='ecfps1', inactives=inact,
    #                                inhibitors=inhib)
    #coevolution_loss_weight('6', 'N(C)(C)C', 'Cramer_2007')
    #violin_plot_weights_in_out('N(C)O', 'Bioorg. Med. Chem. Lett., (2012) 22:6:2242', dset='hERG')
    automatic_violin_two_plots(dset='mutagenicity', model='logreg3', feats='ecfps1', expids=tuple(range(4096)),
                              short_names=False)
    #inact, inhib = map_short_name_to_activity('J. Med. Chem., (2010) 53:1:374', dset='hERG', short_names=False)
    #violin_plot_manager('J. Med. Chem., (2010) 53:1:374', dset='hERG', inactives=inact, inhibitors=inhib,
    #                    feats='ecfps1', model='logreg3', expids=tuple(range(4096)))