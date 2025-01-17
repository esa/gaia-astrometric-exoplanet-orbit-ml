#!/usr/bin/env python
# coding: utf-8

# # Analyse the list of candidates and generate plots and tables for paper
# 
# ### 2023-11-16 - 2024-04-17 Johannes Sahlmann

import logging
import os
    
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
import matplotlib as mp

logger = logging.getLogger()
logger.setLevel(logging.INFO)



overwrite = False
dataset_tag = '0.0.6'
data_path = f'data/{dataset_tag}'
results_path = f'results'

tmp_path = os.path.join(results_path, 'tmp')
os.makedirs(tmp_path, exist_ok=True)

plot_dir = os.path.join(results_path, 'figures')
os.makedirs(plot_dir, exist_ok=True)

candidates_file = os.path.join(results_path, 'candidates.csv')

outfile = os.path.join(data_path, 'labelled_sources.parquet')
labelled_sources = pd.read_parquet(outfile)
logging.info(f"Read {len(labelled_sources)} rows from  {outfile}")


# ## Check that these sources are in the provided dataset
# ### read selected astrometric orbits

outfile = os.path.join(data_path, 'nss_two_body_orbit_astrometric_orbits.parquet')
nss_selected = pd.read_parquet(outfile)

gaia_source = pd.read_parquet(os.path.join(data_path, 'gaia_source_astrometric_orbits.parquet'))
gaia_source = gaia_source.drop_duplicates(subset='source_id')

nss_selected = nss_selected.merge(gaia_source, on='source_id', suffixes=('', '_gaia_source'))
logging.info('Dataset has {} unique source_ids, i.e. {} duplicate source_ids'.format(len(nss_selected['source_id'].unique()), len(nss_selected)-len(nss_selected['source_id'].unique())))

assert np.all(labelled_sources['source_id'].isin(nss_selected['source_id']).values)


# ## remove false positive orbits

remove_false_positives = True
if remove_false_positives:
    nss_selected = nss_selected[~nss_selected['source_id'].isin(labelled_sources.query("label == 'false_positive_orbit'")['source_id'].values)]
    labelled_sources = labelled_sources[~labelled_sources['label'].isin(['false_positive_orbit'])]
    logging.info('Dataset has {} unique source_ids, i.e. {} duplicate source_ids'.format(len(nss_selected['source_id'].unique()), len(nss_selected)-len(nss_selected['source_id'].unique())))

new_candidates_all = pd.read_csv(candidates_file)
logging.info(f"Original file contains {len(new_candidates_all)} candidates ({candidates_file})")
assert len(new_candidates_all) == new_candidates_all['source_id'].nunique()

print(new_candidates_all['relative_occurence'].value_counts().sort_index())



new_candidates_all['label'].value_counts()



new_candidates = new_candidates_all[new_candidates_all['source_id']!=0]
new_candidates = new_candidates[new_candidates['relative_occurence'] > 0.125]
logging.info(f"Cut on relative_occurence > 0.125 leaves {len(new_candidates)} candidates")


# In[13]:


new_candidates['label'].value_counts()


# In[14]:


cols = ['source_id', 'label', 'relative_occurence', 'relative_occurence_ssc', 'relative_occurence_nss']


# In[15]:


# keep only substellar candidates
new_candidates = new_candidates[~new_candidates['label'].isin(['very_low_mass_stellar_companion', 'binary_star'])]

new_candidates_ssc = new_candidates[new_candidates['label'].isin(['substellar_companion_candidates', 'better_substellar_companion_candidates'])]
new_candidates_nss = new_candidates[new_candidates['label'].isnull()]
logging.info(f"number of new_candidates_ssc: {len(new_candidates_ssc)}")
logging.info(f"number of new_candidates_nss: {len(new_candidates_nss)}")

new_candidates_ro = new_candidates.query('relative_occurence > 0.5').reset_index(drop=True)
new_candidates_ro_ids = new_candidates_ro['source_id'].values
print(new_candidates_ro[cols])


new_candidates_ro_ssc = new_candidates.query('relative_occurence_ssc > 0.5').reset_index(drop=True)
new_candidates_ro_ssc = new_candidates_ro_ssc[~new_candidates_ro_ssc['source_id'].isin(new_candidates_ro_ids)]
new_candidates_ro_ssc_ids = new_candidates_ro_ssc['source_id'].values
print(new_candidates_ro_ssc[cols])


new_candidates_ro_nss = new_candidates.query('relative_occurence_nss > 0.5').reset_index(drop=True)
new_candidates_ro_nss = new_candidates_ro_nss[~new_candidates_ro_nss['source_id'].isin(new_candidates_ro_ids)]
new_candidates_ro_nss = new_candidates_ro_nss[~new_candidates_ro_nss['source_id'].isin(new_candidates_ro_ssc_ids)]
new_candidates_ro_nss_ids = new_candidates_ro_nss['source_id'].values
print(new_candidates_ro_nss[cols])


logging.info(f"Cleaned new candidates number: {len(new_candidates)}")
logging.info(f"number of new_candidates_ro: {len(new_candidates_ro)}")
logging.info(f"number of new_candidates_ro_ssc: {len(new_candidates_ro_ssc)}")
logging.info(f"number of new_candidates_ro_nss: {len(new_candidates_ro_nss)}")


new_candidates_all_ids = np.hstack([new_candidates_ro_ids, new_candidates_ro_ssc_ids, new_candidates_ro_nss_ids])
assert len(new_candidates_all_ids) == len(np.unique(new_candidates_all_ids))
len(new_candidates_all_ids)


# In[16]:

# new_candidates_all['source_id'].isin(nss_selected['source_id']).value_counts()
# new_candidates_all[new_candidates_all['source_id'].isin(nss_selected['source_id'])==False]


# In[18]:


from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes


# In[19]:


sel1 = nss_selected.copy()
sel2 = nss_selected[nss_selected['source_id'].isin(labelled_sources[~labelled_sources['label'].isin(['very_low_mass_stellar_companion', 'binary_star', 'false_positive_orbit'])]['source_id'])]
sel3 = sel2.merge(labelled_sources, on='source_id')
sel4 = sel3[sel3['label'].isin(['brown_dwarf_companion', 'exoplanet'])]
sel5 = nss_selected[nss_selected['source_id'].isin(new_candidates)]
sel7 = nss_selected[nss_selected['source_id'].isin(new_candidates_ssc)]
sel8 = nss_selected[nss_selected['source_id'].isin(new_candidates_nss)]
sel9 = nss_selected[nss_selected['source_id'].isin(new_candidates_ro)]
sel10 = nss_selected[nss_selected['source_id'].isin(new_candidates_ro_ssc)]
sel11 = nss_selected[nss_selected['source_id'].isin(new_candidates_all_ids)]
sel12 = sel3[sel3['label'].isin(['exoplanet'])]
sel13 = sel3[sel3['label'].isin(['brown_dwarf_companion'])]

title = None

colour_by = 'bp_rp'
# colour_by = 'mass_function_msun'
norm = mp.colors.LogNorm()
colormap='rainbow'


plot_option = 'all'
# plot_option = 'nss-only'
# plot_option = 'ssc'
# plot_option = 'confirmed'
# plot_option = 'confirmed+candidate'

for x_col in ['bp_rp', 'mass_function_msun']:
    y_col = 'absolute_phot_g_mean_mag'


    from collections import OrderedDict
    datasets = OrderedDict()
    datasets[0] = {'label': f'all ({len(sel1)})', 'data': sel1, 'color': 'k'}
    datasets[1] = {'label': f'substellar candidates ({len(sel2)})', 'data': sel2, 'color': '0.7'}
    datasets[2] = {'label': f'confirmed BD-companions and exoplanets ({len(sel4)})', 'data': sel4, 'color': 'r'}
    datasets[6] = {'label': f'new_candidates_ro ({len(sel9)})', 'data': sel9, 'color': 'b'}
    datasets[7] = {'label': f'new_candidates_ro_ssc ({len(sel10)})', 'data': sel10, 'color': 'g'}
    datasets[8] = {'label': f'Substellar contenders ({len(sel11)})', 'data': sel11, 'color': 'k'}
    datasets[12] = {'label': f'Confirmed exoplanets ({len(sel12)})', 'data': sel12, 'color': 'k'}
    datasets[13] = {'label': f'Confirmed BD-companions ({len(sel13)})', 'data': sel13, 'color': 'k'}


    fig = pl.figure(figsize=(7,4))
    x = datasets[1]['data'][x_col]
    y = datasets[1]['data'][y_col]
    if x_col == 'mass_function_msun':
        bins = [np.logspace(-8, -1, 30), np.linspace(0, 15, 30)]
    else:
        bins=50

    counts,xbins,ybins,image = pl.hist2d(x,y,bins=bins)
    pl.close()

    # for plot_option in ['all', 'nss-only', 'ssc', 'confirmed', 'confirmed+candidate', 'ssc+confirmed']:
    for plot_option in ['ssc+confirmed']:

        fig = pl.figure(figsize=(7,4))
        ax = pl.gca()

        colormap = "Spectral"
        if x_col == 'bp_rp':
            datasets[0]['data'].plot(x_col, y_col, kind='hexbin', gridsize=(200, 100), ax=ax, label=datasets[0]['label'], bins='log', cmap=colormap, colorbar=False)#, cmap="Greys")#, **kwargs)  # , c='parallax'

        elif x_col == 'mass_function_msun':
            df = datasets[0]['data']
            x = df[x_col].values
            y = df[y_col].values
            bins2 = [np.logspace(-7.1, 0.5, 200), np.linspace(-5, 20, 100)]
            H, xedges, yedges = np.histogram2d(x,y,bins=bins2)
            H = np.rot90(H)
            H = np.flipud(H)
            Hmasked = np.ma.masked_where(H==0,H) # Mask pixels with a value of zero
            Hmasked = np.log10(Hmasked)

            pl.pcolormesh(xedges,yedges,Hmasked,cmap=colormap)
            cbar = pl.colorbar()
            cbar.remove()
            ax.set_xscale('log') 

            if x_col == 'mass_function_msun':
                df = datasets[1]['data']
                x = df[x_col].values
                y = df[y_col].values
                bins2 = [np.logspace(-8, 0, 30), np.linspace(0, 15, 30)]
                H, xe, ye = np.histogram2d(x,y,bins=bins2)
                H = np.rot90(H)
                H = np.flipud(H)
                Hmasked = H
                midpoints = (xe[1:] + xe[:-1])/2, (ye[1:] + ye[:-1])/2
                ax.contour(*midpoints, Hmasked, linewidths=2, colors=['b'], levels = [3], label='substellar companion candidates')

        if plot_option in ['all', 'ssc', 'ssc+confirmed']:
            if x_col == 'bp_rp':
                ax.contour(counts.transpose(),extent=[xbins[0],xbins[-1],ybins[0],ybins[-1]], linewidths=2, colors=['b'], levels = [3], label='substellar companion candidates')



        if plot_option in ['all', 'confirmed', 'confirmed+candidate']:
            datasets[2]['data'].plot(x_col, y_col, kind='scatter', ax=ax, label=datasets[2]['label'], c='k', s=30)#, **kwargs)  # , c='parallax'

        if plot_option in ['all', 'confirmed+candidate', 'ssc+confirmed']:
            x = datasets[12]['data'][x_col]
            y = datasets[12]['data'][y_col]
            pl.plot(x, y, marker='o', mfc='w', mec=datasets[12]['color'], ms=6, label=datasets[12]['label'], ls='None')
            x = datasets[13]['data'][x_col]
            y = datasets[13]['data'][y_col]
            pl.plot(x, y, marker='s', mfc='w', mec=datasets[13]['color'], ms=6, label=datasets[13]['label'], ls='None')

            x = datasets[8]['data'][x_col]
            y = datasets[8]['data'][y_col]
            pl.plot(x, y, marker='X', mfc='w', ms=7, mec='k', label=datasets[8]['label'], ls='None')

        ax.invert_yaxis()

        if x_col == 'mass_function_msun':
            ax.set_xlabel('Astrometric mass function $f_M$ ($M_\mathrm{Sun}$)')
        else:    
            ax.set_xlabel('Gaia colour $G_\mathrm{BP}-G_\mathrm{RP}$ (mag)')
        ax.set_ylabel('Gaia absolute magnitude $M_G$ (mag)')

        if x_col == 'bp_rp':
            axins = zoomed_inset_axes(ax, 2, loc=1) # zoom = 6
            x1, x2, y1, y2 = 0.5, 1.6, 2.5, 7.5
            datasets[0]['data'].query(f"{x_col} > {x1} and {x_col} < {x2} and {y_col} > {y1} and {y_col} < {y2}").plot(x_col, y_col, kind='hexbin', gridsize=(50, 50), ax=axins, label=datasets[0]['label'], bins='log', cmap=colormap, colorbar=False)
            axins.set_xlim(x1, x2)
            axins.set_ylim(y1, y2)
            axins.invert_yaxis()
            axins.set_ylabel('')
            axins.set_xlabel('')
            x = datasets[12]['data'][x_col]
            y = datasets[12]['data'][y_col]
            axins.plot(x, y, marker='o', mfc='w', mec=datasets[12]['color'], ms=6, label=datasets[12]['label'], ls='None')
            x = datasets[13]['data'][x_col]
            y = datasets[13]['data'][y_col]
            axins.plot(x, y, marker='s', mfc='w', mec=datasets[13]['color'], ms=6, label=datasets[13]['label'], ls='None')

            x = datasets[8]['data'][x_col]
            y = datasets[8]['data'][y_col]
            axins.plot(x, y, marker='X', c='k', ms=7, mfc='w', label=datasets[8]['label'], ls='None')

            axins.plot(0.82, 4.67, marker='d', c='k', mfc='y', ms=8, label='Sun', ls='None')
        else:
            ax.plot(1.1e-7, 4.67, marker='d', c='k', mfc='y', ms=8, label='Sun', ls='None')

        
        pl.show()

        saveplot = True
        if saveplot:
            figure_file_name = os.path.join(plot_dir, f'{x_col}_{y_col}_{plot_option}.pdf')
            logging.info('Saving figure to {}'.format(figure_file_name))
            fig.savefig(figure_file_name, transparent=True, bbox_inches='tight', pad_inches=0.05)

        pl.close(fig)



# In[21]:


print(sel11[['source_id', 'bp_rp', 'absolute_phot_g_mean_mag', 'mass_function_msun']].sort_values('bp_rp', ascending=False).to_string())

pd.set_option('display.max_columns', 600)
pd.set_option('display.max_rows', 100)
sel_new_candidates = sel11


reference_source_ids_sel_new_candidates = np.array([1712614124767394816, 5323844651848467968, 1576108450508750208,
        522135261462534528, 5148853253106611200, 1156378820136922880,
       6330529666839726592,  364792020789523584, 2884087104955208064,
       5484481960625470336, 2171489736355655680, 1878822452815621120,
       5773484949857279104, 2540855308890440064, 4545802186476906880,
       1897143408911208832, 2280560705703031552, 3909531609393458688,
       3913728032959687424, 3921176983720146560, 1610837178107032192,
       3067074530201582336])


print(f"{sel_new_candidates['source_id'].isin(reference_source_ids_sel_new_candidates).sum()} out of {len(sel_new_candidates)} sources are in the reference list")





