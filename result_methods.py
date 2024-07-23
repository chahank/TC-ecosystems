"""
Created 2024

@author: Chahan M. Kropf
"""

import math
import cartopy

from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import cm, colors, ticker

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature 

import geopandas as gpd
import pandas as pd
import numpy as np
import copy

from pathlib import Path

mpl.rcParams["savefig.dpi"] = 600
plt.rc('legend', fontsize='medium')

from helper_methods import RES, SCE, GCM, TC_CAT, GCM 
from helper_methods import storm_ecoreg_freqs_filename, storm_ecoreg_freqs_quantile_filename, storm_freqs_filename
from plot_vars import PROJ_ATLANTIC, PROJ_PACIFIC, OCEAN_COLOR, CMAP_AFFECTED, CMAP_VULNERABILITIES, CMAP_NEG, CMAP_POS, CMAP_VULNERABILITIES_AFFECTED, NEW_AFFECTED_COLOR, LAT_BOUNDS, LAT_BOUNDS_PROJ, LAT_TROPICS, LAT_TROPICS_PROJ, LON_BOUNDS, LON_BOUNDS_PROJ, TC_CAT_TO_NAMES, LON_BOUNDS_RISK, CCRS_PACIFIC 

MAX_RP = 250
RESULTS = Path('./Results/')
DATA = Path('./Data/')

MINIMUM_FREQUENCY = 1/20
DEP_QUANT = 0.2 #80% of the area is affected 
RES_QUANT = 0.8 #20% of the area is affected
AFF_QUANT = RES_QUANT
COMP_QUANT = 0.5
THRESHOLDS_AT_RISK = np.round(np.arange(0.05, 0.5+0.001, 0.01), 2)
THRESHOLD_DEF = 0.1

def clip_df(x, threshold=1/MAX_RP):
    if np.isreal(x):
        return x if x >= threshold else 0
    else:
        return x
    
def get_max_abs(v1, v2):
    max_val = max(np.abs(v1), np.abs(v2))
    if max_val == np.abs(v1):
        return v1
    return v2

def round_to_e(x, digits=2):
    return f"{x:.{digits}e}"

def res_dep_minprob(min_prob, res=RES_QUANT, dep=DEP_QUANT):
    return f'{res}_{dep}_{min_prob}'

def storm_freq_map_filename(time, tc_cat, gcm, max_rp=MAX_RP, resolution=RES, scenario=SCE):
    if time == 'present':
        return f'tc_storm_freqs_{time}_{tc_cat}_{resolution}_maxrp{max_rp}'
    if time == 'future':
        return f'tc_storm_freqs_{time}_{gcm}_{scenario}_{tc_cat}_{resolution}_maxrp{max_rp}'
    
def storm_nexus_filename(min_prob, tc_cat, resilience=RES_QUANT, dependence=DEP_QUANT, max_rp=MAX_RP, resolution=RES):
    res_dep_minprob_str = res_dep_minprob(min_prob, res=RES_QUANT, dep=DEP_QUANT)
    return 'ecoreg_nexus_tc_storm_{res_dep_minprob_str}_{tc_cat}_present_{resolution}_maxrp{max_rp}'

def vulnerability_plot_filename(min_prob):
    return f'tc_vul_{np.round(min_prob, 2)}_all.pdf'

def zoom_vulnerability_plot_filename(tc_cat, min_prob):
    return f'tc_vul_{np.round(min_prob, 2)}_{tc_cat}_zoom.pdf'

def cc_plots_filename(model=None, tc_cat='allcat'):
    return f'climate_{model}_change_risk_{tc_cat}'

def get_freqs_ecoreg_gdf(time, gcm, resolution=RES, scenario=SCE):
    tc_freq_file_path = DATA / Path(storm_ecoreg_freqs_filename(time, gcm, resolution, scenario)).with_suffix('.shz')
    return gpd.read_file(tc_freq_file_path)

def get_freqs_quantile_ecoreg_df(tc_cat, time, gcm, resolution=RES, scenario=SCE):
    filename = storm_ecoreg_freqs_quantile_filename(tc_cat, time, gcm, resolution, scenario)
    freqs_quantile_ecoreg_path = DATA / Path(filename).with_suffix('.csv')
    return pd.read_csv(freqs_quantile_ecoreg_path)

def add_geom(df, geom_gdf, geo_names_column):
    return gpd.GeoDataFrame(df, geometry=geom_gdf.sort_values(geo_names_column).geometry.values)   

def proba_to_rp(x):
    if x > 0: return 1/x
    return x

def add_tropic_lines(ax, crs):
    from shapely import LineString as line
    for latitude in [60, -60]:
        line_geometry = gpd.GeoSeries(line([[-180, latitude], [180, latitude]]), crs='EPSG:4326')
        line_geometry = line_geometry.to_crs(crs)
        line_geometry.plot(ax=ax, color='red', linewidth=1, linestyle='--')
    for latitude in [40, -40]:
        line_geometry = gpd.GeoSeries(line([[-180, latitude], [180, latitude]]), crs='EPSG:4326')
        line_geometry = line_geometry.to_crs(crs)
        line_geometry.plot(ax=ax, color='gray', linewidth=1, linestyle='--')
        
def add_tropic_lines_proj(ax, crs, hatch='.'):
    from shapely import LineString as line
    ax.fill_between(LON_BOUNDS_PROJ, [LAT_TROPICS_PROJ[0], LAT_TROPICS_PROJ[0]], [LAT_BOUNDS_PROJ[0], LAT_BOUNDS_PROJ[0]], hatch=hatch, alpha=0.2, facecolor="none", edgecolor="darkgreen", linewidth=1.0)
    ax.fill_between(LON_BOUNDS_PROJ, [LAT_TROPICS_PROJ[1], LAT_TROPICS_PROJ[1]], [LAT_BOUNDS_PROJ[1], LAT_BOUNDS_PROJ[1]], hatch=hatch, alpha=0.2, facecolor="none", edgecolor="darkgreen", linewidth=1.0)

def plot_present_frequency_maps(
    tc_cat,
    freqs_ecoreg_gdf,
    ecoreg_geom_plot_atlantic,
    oceans_plot_atlantic,
    proj=PROJ_ATLANTIC,
    label='a',
    time='present',
    gcm='',
    resolution=RES,
    scenario=SCE,
    max_rp=MAX_RP,
    to_file=False
    ):
    
    fig, ax = plt.subplots(figsize=(30,10))
    cmap = 'viridis'
    ecoreg_geom_plot_atlantic.geometry.plot(ax=ax, color='lightgrey')
    oceans_plot_atlantic.plot(ax=ax, color=OCEAN_COLOR)
    gdf = freqs_ecoreg_gdf[freqs_ecoreg_gdf[tc_cat] >= 1/max_rp]
    gdf[tc_cat] = gdf[tc_cat].apply(proba_to_rp)
    gdf_plot = gdf.to_crs(proj)
    gdf_plot.plot(tc_cat, ax=ax, legend=True, markersize=2)
    add_tropic_lines(ax, crs=proj)
    ax.axis('off')
    cax = plt.gcf().axes[1]
    cax.set_ylabel('Average return period [years]', fontsize=20)
    cax.tick_params(axis='both', labelsize=20)
    ax.text(x= -16637797, y=0.9*6935925, s=f'{label})', fontsize=20)
    fig.tight_layout()
    if to_file:
        fig.savefig(
            RESULTS / Path(storm_freq_map_filename(
                time,
                tc_cat,
                gcm,
                max_rp=max_rp,
                resolution=resolution,
                scenario=scenario
            )).with_suffix('.pdf'),
            bbox_inches='tight'
        )

def plot_future_frequency_maps(
    tc_cat,
    freqs_ecoreg_dict,
    ecoreg_geom_plot_atlantic,
    oceans_plot_atlantic,
    proj=PROJ_ATLANTIC,
    label='a',
    time='future',
    gcm='all',
    resolution=RES,
    scenario=SCE,
    max_rp=MAX_RP,
    to_file=False
    ):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(30,10))
    cmap = 'viridis'
    n=0
    for ax, (model, freqs_ecoreg_gdf) in zip(axes.flatten(), freqs_ecoreg_dict.items()):
        ecoreg_geom_plot_atlantic.geometry.plot(ax=ax, color='lightgrey')
        oceans_plot_atlantic.plot(ax=ax, color=OCEAN_COLOR)
        gdf = freqs_ecoreg_gdf[freqs_ecoreg_gdf[tc_cat] >= 1/max_rp]
        gdf[tc_cat] = gdf[tc_cat].apply(proba_to_rp)
        gdf_plot = gdf.to_crs(proj)
        if n == 0:
            ax.text(x= -16637797, y=0.9*6935925, s=f'{label})', fontsize=20)
        if n==3:
            legend=True
        elif n==1:
            legend=True
        else:
            legend=False
        gdf_plot.plot(tc_cat, ax=ax, legend=legend, markersize=2)
        add_tropic_lines(ax, crs=proj)
        ax.axis('off')
        ax.set_title(f'{model}')
        n+=1
    fig.tight_layout()
    if to_file:
        fig.savefig(
            RESULTS / Path(storm_freq_map_filename(
                time,
                tc_cat,
                gcm,
                max_rp=max_rp,
                resolution=resolution,
                scenario=scenario
            )).with_suffix('.pdf'),
            bbox_inches='tight'
        )

'''
Vulnerabilities and CC
'''

categories_dict = {
    'D': 'Dependent',
    'R': 'Resilient',
    'V': 'Vulnerable'
}
categories_dict2 = {
    'D': 'Dependent',
    'R': 'Resilient',
    'VA': 'Vulnerable (affected)',
    'V': 'Vulnerable (not affected)'
}

def map_names(x):
     return categories_dict[x]
    
def vulnerability_nexus(min_prob, time, gcm, dependence_quant=DEP_QUANT, resilience_quant=RES_QUANT, max_rp=MAX_RP, resolution=RES, scenario=SCE, to_file=False):
    nexus_dict = {}
    for tc_cat in TC_CAT:
        filename = RESULTS / Path(storm_nexus_filename(
            min_prob,
            tc_cat,
            resilience=resilience_quant,
            dependence=dependence_quant,
            max_rp=max_rp,
            resolution=resolution
        )).with_suffix('.csv.')
        if filename.is_file():
            nexus = pd.read_csv(filename)
        else:
            freqs_quant = get_freqs_quantile_ecoreg_df(tc_cat, time, gcm, resolution, scenario)
            nexus = freqs_quant.copy(deep=True)
            nexus.insert(0, 'Vulnerability', 'V')
            nexus['Vulnerability'][nexus[str(resilience_quant)] >= min_prob] = 'R' 
            nexus['Vulnerability'][nexus[str(dependence_quant)] >= min_prob] = 'D'
            if to_file:
                nexus.to_csv(filename)
        nexus_dict[tc_cat] = nexus
    return nexus_dict

def nexus_to_affected_map(nexus, affected_quant=AFF_QUANT, max_rp=MAX_RP):
    nexus_copy = copy.deepcopy(nexus)
    nexus_copy['Vulnerability'][(nexus[str(affected_quant)] > 1/max_rp) & (nexus['Vulnerability']=='V')] = 'VA'
    return nexus_copy

def nexus_dict_to_affected(nexus_dict, affected_quant=AFF_QUANT, max_rp=MAX_RP):
    nexus = copy.deepcopy(nexus_dict)
    for tc_cat in TC_CAT:
        nexus[tc_cat] = nexus_to_affected_map(nexus_dict[tc_cat], affected_quant, max_rp)
    return nexus


def plot_affected_regions(ecoreg_geom_plot, min_prob, ecoreg_nexus, affected_quant=AFF_QUANT, figsize=(30, 10), cmap_affected=CMAP_AFFECTED, max_rp=MAX_RP, to_file=True):
    gdf_affected_ecoreg = gpd.GeoDataFrame(
        {'ECO_NAME' : ecoreg_geom_plot.ECO_NAME},
        geometry = ecoreg_geom_plot.geometry
    )
    gdf_affected_ecoreg['affected'] = 0
    gdf_affected_ecoreg['affected'][(ecoreg_nexus['TC1'][affected_quant] > 1/max_rp).values] = 1
    min_lon, min_lat, max_lon, max_lat = gdf_affected_ecoreg[gdf_affected_ecoreg['affected']==1].total_bounds
    fig, ax = plt.subplots(figsize=figsize)
    gdf_affected_ecoreg.plot(ax=ax, column='affected', cmap=cmap_affected, edgecolor='none')
    ax.set_xlim([min_lon, max_lon])
    ax.set_ylim([min_lat, max_lat])
    ax.axis('off')
    fig.set_facecolor(OCEAN_COLOR)
    fig.tight_layout()
    if to_file:
        fig.savefig(RESULTS / f'tc_all_affected_ecosystems_{min_prob}_{affected_quant}.pdf', bbox_inches='tight')

def plot_affected_regions_sensitivity(ecoreg_geom_plot, rp_thres, affected_quants, min_prob=0.2, figsize=(30, 10), cmap_affected=CMAP_AFFECTED, max_rp=MAX_RP, tc_cat='TC1', to_file=True):
    gdf_affected_ecoreg = gpd.GeoDataFrame(
        {'ECO_NAME' : ecoreg_geom_plot.ECO_NAME},
        geometry = ecoreg_geom_plot.geometry
    ) 
    fig, axes = plt.subplots(nrows=len(affected_quants), ncols=len(rp_thres), figsize=figsize)    
    ecoreg_nexus = vulnerability_nexus(
        min_prob=min_prob,
        time='present',
        gcm='',
        to_file=False
    )
    for ax_row, affected_quant in zip(axes, affected_quants):
        for ax, max_rp in zip(ax_row, rp_thres):
            gdf_affected_ecoreg['affected'] = 0
            gdf_affected_ecoreg['affected'][(ecoreg_nexus[tc_cat][str(affected_quant)] > 1/max_rp).values] = 1
            gdf_affected_ecoreg.plot(ax=ax, column='affected', cmap=cmap_affected, edgecolor='none')
            min_lon, min_lat, max_lon, max_lat = gdf_affected_ecoreg[gdf_affected_ecoreg['affected']==1].total_bounds
            ax.set_facecolor(OCEAN_COLOR)
            ax.set_xlim([min_lon, max_lon])
            ax.set_ylim([min_lat, max_lat])
            ax.axis('off')
            ax.set_title(f'max rp:{np.round(max_rp, 2)}, affected area:{np.round(1-affected_quant, 2)}')
            print(affected_quant, max_rp)
    fig.set_facecolor(OCEAN_COLOR)
    fig.tight_layout()
    if to_file:
        fig.savefig(RESULTS / f'tc_all_affected_ecosystems_sensitivity_{tc_cat}.pdf', bbox_inches='tight')
        
def max_nexus(nexus, apply_names=True):
    r3 = nexus['TC3'][nexus['TC3']['Vulnerability'] == 'R']
    r3['Vulnerability'] = 'R'
    d3 = nexus['TC3'][nexus['TC3']['Vulnerability'] == 'D']
    d3['Vulnerability'] = 'D'

    r2 = nexus['TC2'][nexus['TC2']['Vulnerability'] == 'R']
    r2['Vulnerability'] = 'R'
    d2 = nexus['TC2'][nexus['TC2']['Vulnerability'] == 'D']
    d2['Vulnerability'] = 'D'

    r1 = nexus['TC1'][nexus['TC1']['Vulnerability'] == 'R']
    r1['Vulnerability'] = 'R'
    d1 = nexus['TC1'][nexus['TC1']['Vulnerability'] == 'D']
    d1['Vulnerability'] = 'D'
    v1 = nexus['TC1'][nexus['TC1']['Vulnerability'] == 'V']
    v1['Vulnerability'] = 'V'

    df_vul_2 = pd.concat([r3,d3,r2,d2,r1,d1, v1])
    df_vul_2 = df_vul_2.drop_duplicates('ECO_NAME')
    if apply_names:
        df_vul_2['Vulnerability'] = df_vul_2['Vulnerability'].apply(map_names)
    return df_vul_2
        
        
def vulnerability_plots(
    nexus, min_prob, ecoreg_geom, figsize=(30, 10), cmap=CMAP_VULNERABILITIES_AFFECTED,
    ylims=LAT_BOUNDS_PROJ, xlims=LON_BOUNDS_PROJ, inset=True, to_file=False
    ):
    nexus_plot = max_nexus(nexus, apply_names=False)
    nexus_plot = nexus_to_affected_map(nexus_plot)
    gdf_max_vulnerabilities = gpd.GeoDataFrame(
        nexus_plot.sort_values('ECO_NAME'),
        geometry=ecoreg_geom.sort_values('ECO_NAME').geometry.values
    )
    gdf_max_vulnerabilities['Vulnerability'] = gdf_max_vulnerabilities['Vulnerability'].map(categories_dict2)
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': CCRS_PACIFIC})
    gdf_max_vulnerabilities.plot(column='Vulnerability', ax=ax, legend=True, cmap=cmap, categories=categories_dict2.values(), legend_kwds={'fontsize':20}, edgecolor='none')
    #fig.set_facecolor(OCEAN_COLOR)
    if inset:
        inset_ax, x, y, w, h = add_inset(ax, ecoreg_geom)
        df_subset = gdf_max_vulnerabilities.cx[x:x+w, y:y+h]
        df_subset.plot(ax=inset_ax, column='Vulnerability', cmap=cmap, categories=categories_dict2.values(), edgecolor='none')
        ax.indicate_inset_zoom(inset_ax)

        inset_ax, x, y, w, h = add_inset2(ax, ecoreg_geom)
        df_subset = gdf_max_vulnerabilities.cx[x:x+w, y:y+h]
        df_subset.plot(ax=inset_ax, column='Vulnerability', cmap=cmap, categories=categories_dict2.values(), edgecolor='none')
        ax.indicate_inset_zoom(inset_ax)
        
    ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
    add_tropic_lines_proj(ax, crs=ecoreg_geom.crs)
    ax.add_feature(cfeature.OCEAN, color=OCEAN_COLOR)
    ax.add_feature(cfeature.LAKES, alpha=0.5)
    ax.set_ylim(ylims)
    ax.set_xlim(xlims)
    fig.tight_layout()
    if to_file:
        fig.savefig(RESULTS / Path(vulnerability_plot_filename(min_prob)).with_suffix('.pdf'), bbox_inches='tight')
        
        
def vulnerability_zoom_plots(min_prob, ecoreg_geom, ylims=LAT_BOUNDS_PROJ, xlims=LON_BOUNDS_PROJ, figsize=(30, 10), cmap=CMAP_VULNERABILITIES_AFFECTED,
    inset=True, to_file=False):
    nexus = vulnerability_nexus(
            min_prob=min_prob,
            time='present',
            gcm='',
            to_file=False
        )
    gdf_plot_dict = {}
    for cat in TC_CAT:
        reduce_ecoreg = nexus_to_affected_map(nexus[cat])[['ECO_NAME', 'Vulnerability']]
        reduce_ecoreg['Vulnerability'] = reduce_ecoreg['Vulnerability'].map(categories_dict2)
        gdf_plot = gpd.GeoDataFrame(reduce_ecoreg.sort_values('ECO_NAME'), geometry=ecoreg_geom.sort_values('ECO_NAME').geometry.values)
        gdf_plot_dict[cat] = gdf_plot
    slist = ['a)', 'b)', 'c)']
    
    for s, cat in zip(slist, TC_CAT):
        fig, ax = plt.subplots(figsize=figsize,  subplot_kw={'projection': CCRS_PACIFIC})
        if gdf_plot_dict[cat].shape[0] == 0:
            print('no resilience')
            continue
        gdf_plot_dict[cat].plot(column='Vulnerability', ax=ax, legend=True, cmap=cmap, categories=categories_dict2.values(), edgecolor='none', legend_kwds={'fontsize':20})
        if inset:
            inset_ax, x, y, w, h = add_inset(ax, ecoreg_geom)
            df_subset = gdf_plot_dict[cat].cx[x:x+w, y:y+h]
            df_subset.plot(ax=inset_ax, column='Vulnerability', cmap=cmap, categories=categories_dict2.values(), edgecolor='none')
            ax.indicate_inset_zoom(inset_ax)

            inset_ax, x, y, w, h = add_inset2(ax, ecoreg_geom)
            df_subset = gdf_plot_dict[cat].cx[x:x+w, y:y+h]
            df_subset.plot(ax=inset_ax, column='Vulnerability', cmap=cmap, categories=categories_dict2.values(), edgecolor='none')
            ax.indicate_inset_zoom(inset_ax)
        
        ax.text(x=xlims[0] * 0.9, y=ylims[1]*0.9, s=s, fontsize=20)
        ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
        add_tropic_lines_proj(ax, crs=ecoreg_geom.crs)
        ax.add_feature(cfeature.OCEAN, color=OCEAN_COLOR)
        ax.add_feature(cfeature.LAKES, alpha=0.5)
        ax.set_ylim(ylims)
        ax.set_xlim(xlims)
        fig.tight_layout()
        if to_file:
            filename = RESULTS / Path(zoom_vulnerability_plot_filename(cat, min_prob)).with_suffix('.pdf')
            fig.savefig(filename, bbox_inches='tight')  

def vulnerability_sensitivity_plots(
    min_probs, ecoreg_geom, figsize=(30, 20), cmap=CMAP_VULNERABILITIES_AFFECTED,
    ylims=[-3.8e6, 6e6], xlims=[-1e7, 1.3e7], to_file=False
    ):
    fig, axes = plt.subplots(ncols=len(min_probs), nrows=4, figsize=figsize)
    for ax_rows, min_prob in zip(axes.T, min_probs): 
        nexus = vulnerability_nexus(
                    min_prob=min_prob,
                    time='present',
                    gcm='',
                    to_file=False
                )
        for ax, cat in zip(ax_rows, ['all', 'TC1', 'TC2', 'TC3']):
            print(cat)
            if cat == 'all':
                nexus_plot = max_nexus(nexus, apply_names=False)
            else:
                nexus_plot = nexus[cat]
            nexus_plot = nexus_to_affected_map(nexus_plot)
            gdf_vulnerabilities = gpd.GeoDataFrame(
                nexus_plot.sort_values('ECO_NAME'),
                geometry=ecoreg_geom.sort_values('ECO_NAME').geometry.values
            )
            gdf_vulnerabilities['Vulnerability'] = gdf_vulnerabilities['Vulnerability'].map(categories_dict2)
            gdf_vulnerabilities.plot(column='Vulnerability', ax=ax, cmap=cmap, categories=categories_dict2.values(), edgecolor='none')
            ax.set_ylim(ylims)
            ax.set_xlim(xlims)
            ax.axis('off')
            ax.set_title(f'fmin: {np.round(min_prob,2)} ; cat: {cat}')
    fig.set_facecolor(OCEAN_COLOR)
    fig.tight_layout()
    if to_file:
        fig.savefig(RESULTS / Path(vulnerability_plot_filename(min_prob) + '_sensitivity').with_suffix('.pdf'), bbox_inches='tight')

'''
Climate change plots
'''

def ecoreg_diff_cc(nexus, ecoreg_geom, max_rp=MAX_RP, resilience_quant=RES_QUANT, dependence_quant=DEP_QUANT,
                  resolution=RES, scenario=SCE):
    ecoreg_diff = {}
    for tc_cat in TC_CAT:
        df = pd.DataFrame(nexus[tc_cat][['ECO_NAME', 'Vulnerability']])
        df1 = nexus[tc_cat].copy()
        df['Present'] = None
        select = df1['Vulnerability'] == 'V'
        df['Present'][select] = df1[select][str(resilience_quant)]
        select = df1['Vulnerability'] == 'R'
        df['Present'][select] = df1[select][str(resilience_quant)]
        select = df1['Vulnerability'] == 'D'
        df['Present'][select] = df1[select][str(dependence_quant)]
        for model in GCM:
            quantiles_cc = get_freqs_quantile_ecoreg_df(
                tc_cat, 'future', resolution=resolution, gcm=model, scenario=scenario)
            df2 = quantiles_cc.copy()
            df[model] = None
            select = df1['Vulnerability'] == 'V'
            df[model][select] = df2[select][str(resilience_quant)]
            select = df1['Vulnerability'] == 'R'
            df[model][select] = df2[select][str(resilience_quant)]
            select = df1['Vulnerability'] == 'D'
            df[model][select] = df2[select][str(dependence_quant)]
        df['ModelMean'] = df[df.columns[3:]].mean(axis=1)
        df['ModelMedian'] = df[df.columns[3:-1]].median(axis=1)
        df = df.applymap(lambda x: clip_df(x, 1/max_rp))
        for m in ['Mean', 'Median']:
            df[f'Model{m}Diff'] = df[f'Model{m}'] - df['Present']
            df[f'Model{m}RelDiff'] = (df[f'Model{m}'] - df['Present']) / (np.abs(df['Present']))
        for gcm in GCM:
            df[f'Model{gcm}Diff'] = df[f'{gcm}'] - df['Present']
            df[f'Model{gcm}RelDiff'] = (df[f'{gcm}'] - df['Present']) / (np.abs(df['Present']))
            
        df[['BIOME_NAME', 'REALM', 'SHAPE_AREA']] = ecoreg_geom[['BIOME_NAME', 'REALM', 'SHAPE_AREA']].values
        ecoreg_diff[tc_cat] = df   
    return ecoreg_diff

def ecoreg_diff_at_risk(ecoreg_diff_cc, thresholds=THRESHOLDS_AT_RISK, model='Median'):
    sig_diff = {}
    for threshold in thresholds:
        dic = {}
        for tc_cat in TC_CAT:
            df = ecoreg_diff_cc[tc_cat].dropna()
            select = (
            (df['Vulnerability'] == 'V') & (df[f'Model{model}RelDiff'] >= threshold) |
            (df['Vulnerability'] == 'R') & (df[f'Model{model}RelDiff'] >= 2*threshold) |
            (df['Vulnerability'] == 'D') & (df[f'Model{model}RelDiff'].abs() >= threshold/2))
            dic[tc_cat] = df[select]
        sig_diff[threshold] = dic
    return sig_diff

def prepare_cc_data_to_plot(sig_diff, ecoreg_geom, threshold_sig_diff=THRESHOLD_DEF, model='Median', tc_cat='all'):
    
    col = f'Model{model}RelDiff'
    thres = threshold_sig_diff
    
    if tc_cat in TC_CAT:
        df = sig_diff[thres][tc_cat]
        return gpd.GeoDataFrame({'MaxChange': df[col].values}, geometry=ecoreg_geom[np.isin(ecoreg_geom['ECO_NAME'], [df.ECO_NAME])].geometry.values)
    elif tc_cat!='all':
        return None
    
    df = sig_diff[thres]['TC1']
    gdf1 = gpd.GeoDataFrame(df, geometry=ecoreg_geom[np.isin(ecoreg_geom['ECO_NAME'], [df.ECO_NAME])].geometry.values)
    data1 = gdf1.loc[:,['geometry', col]]
    df = sig_diff[thres]['TC2']
    gdf2 = gpd.GeoDataFrame(df, geometry=ecoreg_geom[np.isin(ecoreg_geom['ECO_NAME'], [df.ECO_NAME])].geometry.values)
    data2 = gdf2.loc[:,['geometry', col]]

    index1 = data1.index.values
    val1 = data1[col].values
    geom1 = data1.geometry.values
    index2 = data2.index.values
    val2 = data2[col].values
    geom2 = data2.geometry.values

    indices = np.unique(np.hstack([index1, index2]))
    data_list = []
    geom_list = []
    index_list = []
    for idx in indices:
        if np.isin(idx, index1):
            a1 = np.argwhere(index1 == idx)[0][0]
            v1 = val1[a1]
            geom = geom1[a1]
        else:
            v1 = 0
        if np.isin(idx, index2):
            a2 = np.argwhere(index2 == idx)[0][0]
            v2 = val2[a2]
            geom = geom2[a2]
        else:
            v2 = 0
        index_list.append(idx)
        data_list.append(get_max_abs(v1, v2))
        geom_list.append(geom)
    gdf_max = gpd.GeoDataFrame({"MaxChange": data_list}, geometry=geom_list, index=index_list)


    df = sig_diff[thres]['TC3']
    gdf2 = gpd.GeoDataFrame(df, geometry=ecoreg_geom[np.isin(ecoreg_geom['ECO_NAME'], [df.ECO_NAME])].geometry.values)
    data2 = gdf2.loc[:,['geometry', col]]
    data1 = gdf_max.loc[:,['geometry', 'MaxChange']]

    index1 = data1.index.values
    val1 = data1.MaxChange.values
    geom1 = data1.geometry.values
    index2 = data2.index.values
    val2 = data2[col].values
    geom2 = data2.geometry.values

    indices = np.unique(np.hstack([index1, index2]))
    data_list = []
    geom_list = []
    index_list = []
    for idx in indices:
        if np.isin(idx, index1):
            a1 = np.argwhere(index1 == idx)[0][0]
            v1 = val1[a1]
            geom = geom1[a1]
        else:
            v1 = 0
        if np.isin(idx, index2):
            a2 = np.argwhere(index2 == idx)[0][0]
            v2 = val2[a2]
            geom = geom2[a2]
        else:
            v2 = 0
        index_list.append(idx)
        data_list.append(get_max_abs(v1, v2))
        geom_list.append(geom)
    gdf_max = gpd.GeoDataFrame({"MaxChange": data_list}, geometry=geom_list, index=index_list)
    return gdf_max


def plot_cc_sensitivity(
    min_probs,  ecoreg_geom_plot, ecoreg_atlantic, max_rp=MAX_RP,
    thresholds_sig_diff=[THRESHOLD_DEF], figsize=(30,20), model='Median', tc_cat='allcat',
    to_file=True, **kwargs):
    fig, axes = plt.subplots(nrows=len(thresholds_sig_diff), ncols=len(min_probs), figsize=figsize, subplot_kw={'projection': CCRS_PACIFIC})
    for m, (ax_cols, min_prob) in enumerate(zip(axes, min_probs)):
        nexus_present = vulnerability_nexus(
            min_prob=min_prob,
            time='present',
            gcm='',
            to_file=False,
            max_rp=max_rp
        )
        ecoreg_diff = ecoreg_diff_cc(nexus_present, ecoreg_atlantic, max_rp=max_rp)
        ecoreg_at_risk = ecoreg_diff_at_risk(ecoreg_diff, model=model)
        for ax, threshold_sig_diff in zip(np.array(ax_cols).flatten(), thresholds_sig_diff):
            legend=True
            plot_cc(ecoreg_at_risk, min_prob=min_prob, tc_cat=tc_cat, ecoreg_geom=ecoreg_geom_plot, threshold_sig_diff=threshold_sig_diff, ax=ax, xlabel=False, to_file=False, legend=legend, **kwargs)
            ax.set_title(f"minprob={np.round(min_prob,2)}, risk_thres={np.round(threshold_sig_diff,2)}");
    fig.tight_layout()
    if to_file:
        filename = RESULTS / Path(cc_plots_filename(model, tc_cat) + '_sensitivity').with_suffix('.pdf')
        fig.savefig(filename, bbox_inches='tight')
        
def plot_cc_gcm_sensitivity(
    thresholds_sig_diff, models, ecoreg_geom_plot, ecoreg_atlantic, max_rp=MAX_RP,
    min_prob=1/20, figsize=(30,20), tc_cat='all',
    to_file=True, **kwargs):
    plt.style.use('default')
    fig, axes = plt.subplots(nrows=len(models), ncols=len(thresholds_sig_diff), figsize=figsize, subplot_kw={'projection': CCRS_PACIFIC})
    for m, (ax_cols, model) in enumerate(zip(axes, models)):
        nexus_present = vulnerability_nexus(
            min_prob=min_prob,
            time='present',
            gcm='',
            to_file=False,
            max_rp=max_rp
        )
        ecoreg_diff = ecoreg_diff_cc(nexus_present, ecoreg_atlantic, max_rp=max_rp)
        for n, (ax, threshold_sig_diff) in enumerate(zip(np.array(ax_cols).flatten(), thresholds_sig_diff)):
            print(model, threshold_sig_diff)
            ecoreg_at_risk = ecoreg_diff_at_risk(ecoreg_diff, model=model)
            if m == len(models)-1:
                legend = True
                force_neg_scale = True
                xlabel=False
            else:
                legend = False
                force_neg_scale = False
                xlabel=False
            plot_cc(ecoreg_at_risk, min_prob=min_prob, tc_cat=tc_cat, ecoreg_geom=ecoreg_geom_plot, threshold_sig_diff=threshold_sig_diff, model=model, ax=ax, xlabel=xlabel, to_file=False, legend=legend, force_neg_scale=force_neg_scale, **kwargs)
            if m == 0:
                ax.set_title(f"Risk threshold: {np.round(threshold_sig_diff,2)}", fontsize=14, pad=10)
            if n == 0 :
                ax.annotate(
                    f"{model}", xy=(-0.05, 0.5), xycoords='axes fraction',
                    verticalalignment='center', horizontalalignment='left',
                    fontsize=14, rotation=90
                )
    fig.tight_layout()
    if to_file:
        filename = RESULTS / Path(cc_plots_filename('all_models', tc_cat) + '_sensitivity').with_suffix('.pdf')
        fig.savefig(filename, bbox_inches='tight')


def add_inset(ax, ecoreg_geom):
    # Inset axis location
    x = 0.5e7
    y = -5.2e6
    w = 1e7
    h = 5e6
    inset = ax.inset_axes([x, y, w, h], transform=ax.transData)
    # Inset data
    x = 0.865e7
    y = 2e6
    w = 0.28e7
    h = 1.5e6
    df_ecoreg = ecoreg_geom.cx[x:x+w, y:y+h]
    df_ecoreg.plot(ax= inset, color='lightgrey', edgecolor='none')
    inset.set_xlim([x, x+w])
    inset.set_ylim([y, y+h])
    inset.set_facecolor(OCEAN_COLOR)
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_xticklabels([])
    inset.set_yticklabels([])
    return inset, x, y, w, h

def add_inset2(ax, ecoreg_geom):
    # Inset axis location
    x = -1.5e7
    y = 4.1e6
    w = 1.3e7
    h = 3e6
    inset = ax.inset_axes([x, y, w, h], transform=ax.transData)
    # Inset data
    x = -0.58e7
    y = 0.65e6
    w = 0.3e7
    h = 1.95e6
    df_ecoreg = ecoreg_geom.cx[x:x+w, y:y+h]
    df_ecoreg.plot(ax= inset, color='lightgrey', edgecolor='none')
    inset.set_xlim([x, x+w])
    inset.set_ylim([y, y+h])
    inset.set_facecolor('azure')
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_xticklabels([])
    inset.set_yticklabels([])
    return inset, x, y, w, h

def plot_cc(sig_diff, min_prob, ecoreg_geom, threshold_sig_diff=THRESHOLD_DEF, figsize=(30, 10), lon_bounds = LON_BOUNDS_RISK,
            model='Median',  tc_cat='all', cmap_pos=CMAP_POS, vmin=5, vmax=250, new_affected_color=NEW_AFFECTED_COLOR, inset=True,
            to_file=False, ax=None, xlabel=True, gridlines=True, legend=True, remove_NAEUAF=False, force_neg_scale=False, axis_off=True):
    
    gdf_max = prepare_cc_data_to_plot(sig_diff, ecoreg_geom, threshold_sig_diff, model=model, tc_cat=tc_cat)
    
    max_prob = gdf_max['MaxChange'].replace([np.inf, -np.inf], np.nan).dropna().max()
    if not vmax:
        vmax = np.round(max_prob*100)
    if not vmin:
        vmin = min_prob*100
    norm_pos = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    max_neg_prob = gdf_max['MaxChange'].replace([np.inf, -np.inf], np.nan).dropna().min()
    norm_neg = mpl.colors.Normalize(vmin=-12, vmax=-5)

    change_ticks = [5, 50, 100, 150, 200, 250]
    neg_change_ticks = [-12, -10, -8, -6]
    
    col = 'MaxChange'
    gdf = gdf_max

    data = gdf.loc[:,['geometry', col]]
    data[col] = data[col]*100
    inf = data.index[np.isinf(data[col])]

    plot_data = data.replace([np.inf, -np.inf], np.nan)
    plot_data.dropna(inplace=True)

    pos_gdf = plot_data[plot_data[col] >= 0]
    neg_gdf = plot_data[plot_data[col] < 0]
    
    gdf_handle = [mpatches.Patch(color='lightgrey')]
    gdf_label = ['not at risk']

    inf_handle = [mpatches.Patch(color=new_affected_color)]
    inf_label = ['newly affected']
    if not ax:
        fig = plt.figure(figsize=(30, 10))
        ax = fig.add_subplot(1, 1, 1, projection=CCRS_PACIFIC)
    else:
        fig = ax.get_figure()
    min_percent = 100 * min_prob
    
    ecoreg_geom.geometry.plot(ax=ax, color='lightgrey', edgecolor='none')
    
    if inset:
        inset_ax1, x1, y1, w1, h1 = add_inset(ax, ecoreg_geom)
        inset_ax2, x2, y2, w2, h2 = add_inset2(ax, ecoreg_geom)
        inset_info = [[inset_ax1, x1, y1, w1, h1], [inset_ax2, x2, y2, w2, h2]]
    
    if len(neg_gdf) != 0:
        neg_gdf.plot(column = col, cmap=CMAP_NEG, ax=ax, norm=norm_neg)
        pos_gdf.plot(column = col, cmap=cmap_pos, ax=ax, norm=norm_pos)
        if inset:
            for inset_ax, x, y, w, h in inset_info:
                df_subset = pos_gdf.cx[x:x+w, y:y+h]
                df_subset.plot(ax=inset_ax, column = col, cmap=CMAP_POS, norm=norm_pos)
                df_subset = neg_gdf.cx[x:x+w, y:y+h]
                df_subset.plot(column = col, cmap=CMAP_NEG, ax=inset_ax, norm=norm_neg)
    else: 
        pos_gdf.plot(column = col, cmap=cmap_pos, ax=ax, norm=norm_pos)
        if inset:
            for inset_ax, x, y, w, h in inset_info:
                df_subset = pos_gdf.cx[x:x+w, y:y+h]
                df_subset.plot(ax=inset_ax, column = col, cmap=CMAP_POS, norm=norm_pos)

    #for i in inf:
    inf_reg = data.loc[inf]
    #    inf_reg = pd.DataFrame(inf_reg).T
    #    inf_reg = gpd.GeoDataFrame(inf_reg, geometry='geometry')
    inf_reg.plot(ax=ax, color =new_affected_color, legend = False)
    if inset:
        for inset_ax, x, y, w, h in inset_info:
            df_subset = inf_reg.cx[x:x+w, y:y+h]
            df_subset.plot(ax=inset_ax, color =new_affected_color, legend = False)
    
    if remove_NAEUAF:
        ecoreg_geom[np.isin(ecoreg_geom['ECO_NAME'], ecoregions_NAEUAF)].plot(ax=ax, color='lightgrey', edgecolor='none')
    
    if legend:
        #ax.set_title(f'Ecosystems affected by Tropical Cyclones changes from Climate Change \n 2050 RCP8.5', fontsize=22)
        handles=inf_handle + gdf_handle
        labels=inf_label + gdf_label
        ax.legend(handles, labels, ncol=5, prop={'size':15}, numpoints=1, loc='lower left')   

        if len(neg_gdf) != 0 or force_neg_scale:
            cax_neg = ax.inset_axes(
                 bounds =(0.21, -0.1, 0.1, 0.05), transform = ax.transAxes
            )
            cbar_neg = fig.colorbar(
                    cm.ScalarMappable(cmap=CMAP_NEG, norm = norm_neg), cax = cax_neg, orientation = 'horizontal'
                )
            cbar_neg.set_ticks(neg_change_ticks)

        cax_pos = ax.inset_axes(bounds =(0.35, -0.1, 0.4, 0.05), transform = ax.transAxes)
        cbar_pos = fig.colorbar(
            cm.ScalarMappable(cmap=cmap_pos, norm = norm_pos), cax = cax_pos, orientation = 'horizontal'
        )
        cbar_pos.set_ticks(change_ticks)
    if xlabel:
        cax_pos.set_xlabel('Relative Change [%] of Tropical Cyclone frequency \n per Ecoregion', fontsize = 22)

    min_lat, max_lat = LAT_BOUNDS_PROJ
    add_tropic_lines_proj(ax, crs=ecoreg_geom.crs)
    ax.add_feature(cfeature.OCEAN, color=OCEAN_COLOR)
    ax.add_feature(cfeature.LAKES, alpha=0.5)
    #ax.set_xlim([-2e7, 2e7])
    ax.set_ylim([min_lat, max_lat])
    if gridlines:
        ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
    if inset:
        for inset_ax, x, y, w, h in inset_info:
            ax.indicate_inset_zoom(inset_ax)
    if lon_bounds:
        min_lon, max_lon = lon_bounds
        ax.set_xlim([min_lon, max_lon])
    if axis_off:
        ax.axis('off')
    fig.tight_layout()
    #fig.set_facecolor(OCEAN_COLOR)
    if to_file:
        filename = RESULTS / Path(cc_plots_filename(model, tc_cat)).with_suffix('.pdf')
        fig.savefig(filename, bbox_inches='tight')
    return ax
        
def gcm_change_agreement(
    ecoreg_atlantic, ecoreg_pacific, min_prob=MINIMUM_FREQUENCY, max_rp=MAX_RP, agree_ratio = 0.75, to_file=True
):
    nexus_present = vulnerability_nexus(
        min_prob=min_prob,
        time='present',
        gcm='',
        to_file=False,
        max_rp=max_rp
    )
    ecoreg_diff = ecoreg_diff_cc(nexus_present, ecoreg_atlantic, max_rp=max_rp)
    gcm_agreement_pos = np.ones(ecoreg_diff['TC1'].shape[0], dtype='bool')
    for tc_cat in TC_CAT[0:1]:
        agreement = (np.mean(ecoreg_diff[tc_cat][[f'Model{model}RelDiff' for model in GCM]].fillna(0).gt(0.1), axis=1) >=agree_ratio).values
        gcm_agreement_pos = np.logical_and(agreement, gcm_agreement_pos)
        agreement = (np.mean(ecoreg_diff[tc_cat][[f'Model{model}RelDiff' for model in GCM]].fillna(0).lt(1e20), axis=1) >=agree_ratio).values
        gcm_agreement_pos = np.logical_and(agreement, gcm_agreement_pos)

    gcm_agreement_new = np.ones(ecoreg_diff['TC1'].shape[0], dtype='bool')
    for tc_cat in TC_CAT[0:1]:
        agreement = (np.mean(ecoreg_diff[tc_cat][[f'Model{model}RelDiff' for model in GCM]].fillna(0).gt(1e20), axis=1) >=agree_ratio).values
        gcm_agreement_new = np.logical_and(agreement, gcm_agreement_new)    

    gcm_agreement_neg = np.ones(ecoreg_diff['TC1'].shape[0], dtype='bool')
    for tc_cat in TC_CAT[0:1]:
        agreement = (np.mean(ecoreg_diff[tc_cat][[f'Model{model}RelDiff' for model in GCM]].fillna(0).lt(0), axis=1) >= agree_ratio).values
        gcm_agreement_neg = np.logical_and(agreement, gcm_agreement_neg)

    gdf_agreement = gpd.GeoDataFrame(
        {'Increase': gcm_agreement_pos,
         'Decrease': gcm_agreement_neg,
         'New': gcm_agreement_new
        },
        geometry = ecoreg_pacific.geometry.values
    )
    from plot_vars import CMAP_AGREEMENT, LAT_BOUNDS_PROJ, OCEAN_COLOR
    fig, axes = plt.subplots(ncols=3, figsize=(30, 20))
    for ax, column in zip(axes, gdf_agreement.columns):
        gdf_agreement.plot(
            column=column, ax=ax, legend=True, cmap=CMAP_AGREEMENT, categories=[True, False],
            edgecolor='none', legend_kwds={'labels':[f'{column} agreement', 'No agreement'],'fontsize':14, 'loc':'lower right'}
        )
        ax.axis('off')
        min_lat, max_lat = LAT_BOUNDS_PROJ
        ax.set_ylim([min_lat, max_lat])
    fig.tight_layout()
    fig.set_facecolor(OCEAN_COLOR)
    if to_file:
        filename = RESULTS / Path(f'GCM_change_agreement_agree{agree_ratio}').with_suffix('.pdf')
        fig.savefig(filename, bbox_inches='tight')
    return fig, ax

'''
Get the numbers
'''

def make_rp_area_table(nexus, ecoreg_geom_area, affected_quant='mean', max_rp=250):
    all_V = pd.DataFrame(ecoreg_geom_area[['ECO_NAME', 'BIOME_NAME', 'REALM', 'SHAPE_AREA']])
    all_V['V TC1'] = nexus['TC1']['Vulnerability']
    all_V['Rp TC1'] = nexus['TC1'][str(affected_quant)].apply(proba_to_rp)
    all_V['Rp TC1'] = all_V['Rp TC1'][all_V['Rp TC1'] <= max_rp]
    all_V['V TC2'] = nexus['TC2']['Vulnerability']
    all_V['Rp TC2'] = nexus['TC2'][str(affected_quant)].apply(proba_to_rp)
    all_V['Rp TC2'] = all_V['Rp TC2'][all_V['Rp TC2'] <= max_rp]
    all_V['V TC3'] = nexus['TC3']['Vulnerability']
    all_V['Rp TC3'] = nexus['TC3'][str(affected_quant)].apply(proba_to_rp)
    all_V['Rp TC3'] = all_V['Rp TC3'][all_V['Rp TC3'] <= max_rp]
    return all_V.fillna(0)

def get_cc_affected_area(
        ecoreg_geom_area, model='Median', threshold_at_risk=0.1, min_prob=0.05,
        max_rp=250
    ):
    nexus_present = vulnerability_nexus(
        min_prob=min_prob,
        time='present',
        gcm='',
        to_file=False,
        max_rp=max_rp
    )
    tot_area = ecoreg_geom_area['SHAPE_AREA'].sum()
    ecoreg_diff = ecoreg_diff_cc(nexus_present, ecoreg_geom_area, max_rp=max_rp)
    ecoreg_at_risk = ecoreg_diff_at_risk(ecoreg_diff, model=model)    
    dfs = []
    for tc_cat in TC_CAT:
        increase = ecoreg_at_risk[threshold_at_risk][tc_cat][
            (ecoreg_at_risk[threshold_at_risk][tc_cat][f'Model{model}RelDiff'] < 10000)\
            & (ecoreg_at_risk[threshold_at_risk][tc_cat][f'Model{model}RelDiff'] >0)
        ]['SHAPE_AREA'].sum()
        new = ecoreg_at_risk[threshold_at_risk][tc_cat][
            ecoreg_at_risk[threshold_at_risk][tc_cat][f'Model{model}RelDiff'] > 10000
        ]['SHAPE_AREA'].sum()
        total = ecoreg_at_risk[threshold_at_risk][tc_cat]['SHAPE_AREA'].sum()
        decrease = np.round(total - increase - new)
        dfs.append(
            pd.DataFrame(
            {f'Ecoregion Area {tc_cat}': [increase, new, decrease, total],
            f'Ecoregion relative total Area {tc_cat} [%]' : np.round([increase/tot_area, new/tot_area, decrease/tot_area, total/tot_area], 3) * 100},
            index = ['increase', 'new', 'decrease', 'total']
            )
        )
    return pd.concat(dfs, axis=1).applymap(round_to_e)

def get_at_risk_ecoregions_counts(
        ecoreg_geom, model='Median', threshold_at_risk=0.1, min_prob=0.05,
        max_rp=250
    ):
    nexus_present = vulnerability_nexus(
        min_prob=min_prob,
        time='present',
        gcm='',
        to_file=False,
        max_rp=max_rp
    )
    ecoreg_diff = ecoreg_diff_cc(nexus_present, ecoreg_geom, max_rp=max_rp)
    ecoreg_at_risk = ecoreg_diff_at_risk(ecoreg_diff, model=model)    
    dfs = []
    for tc_cat in TC_CAT:
        increase = ecoreg_at_risk[threshold_at_risk][tc_cat][
            (ecoreg_at_risk[threshold_at_risk][tc_cat][f'Model{model}RelDiff'] < 10000)\
            & (ecoreg_at_risk[threshold_at_risk][tc_cat][f'Model{model}RelDiff'] >0)
        ].count()['ECO_NAME']
        new = ecoreg_at_risk[threshold_at_risk][tc_cat][
            ecoreg_at_risk[threshold_at_risk][tc_cat][f'Model{model}RelDiff'] > 10000
        ].count()['ECO_NAME']
        total = ecoreg_at_risk[threshold_at_risk][tc_cat].shape[0]
        decrease = total - increase - new
        dfs.append(
            pd.DataFrame(
            {f'Ecoregion Counts {tc_cat}': [increase, new, decrease, total]},
            index = ['increase', 'new', 'decrease', 'total']
            )
        )
    return pd.concat(dfs, axis=1)

def get_change_frequency(ecoreg_atlantic, min_prob=MINIMUM_FREQUENCY, threshold_at_risk=THRESHOLD_DEF, max_rp=MAX_RP):
    import pandas as pd
    nexus_present = vulnerability_nexus(
        min_prob=min_prob,
        time='present',
        gcm='',
        to_file=False,
        max_rp=max_rp
    )
    ecoreg_diff = ecoreg_diff_cc(nexus_present, ecoreg_atlantic, max_rp=max_rp)
    average_change_CC = pd.DataFrame({
        'TC category': TC_CAT,
    })
    for model in GCM + ['Median']:
        ecoreg_at_risk = ecoreg_diff_at_risk(ecoreg_diff, thresholds=np.array([threshold_at_risk]), model=model)
        median_cc = [ecoreg_at_risk[threshold_at_risk][tc_cat][f'Model{model}RelDiff'].median() * 100 for tc_cat in TC_CAT]
        average_change_CC[f'Median CC {model} [%]']= np.round(median_cc,1)
    return average_change_CC

def get_tc_affected_area(ecoreg_geom_area, min_prob, ecoreg_nexus, affected_quant=AFF_QUANT, max_rp=MAX_RP):
    df_affected_ecoreg = pd.DataFrame(
        {'ECO_NAME' : ecoreg_geom_area.sort_values('ECO_NAME').ECO_NAME.values,
         'SHAPE_AREA': ecoreg_geom_area.sort_values('ECO_NAME').SHAPE_AREA.values}
    )
    df_affected_ecoreg['affected_TC1'] = 0 
    df_affected_ecoreg['affected_TC2'] = 0 
    df_affected_ecoreg['affected_TC3'] = 0 

    df_affected_ecoreg['affected_TC1'][(ecoreg_nexus['TC1'][affected_quant] > 1/max_rp).values] = 1
    df_affected_ecoreg['affected_TC2'][(ecoreg_nexus['TC2'][affected_quant] > 1/max_rp).values] = 1
    df_affected_ecoreg['affected_TC3'][(ecoreg_nexus['TC3'][affected_quant] > 1/max_rp).values] = 1
    
    area_total = df_affected_ecoreg.SHAPE_AREA.sum()
    
    results = pd.DataFrame(
        {tc_cat: [
            df_affected_ecoreg[df_affected_ecoreg[f'affected_{tc_cat}'] == 1]['SHAPE_AREA'].sum(),
            df_affected_ecoreg[df_affected_ecoreg[f'affected_{tc_cat}'] == 1]['SHAPE_AREA'].sum() / area_total,
            area_total,
            df_affected_ecoreg[f'affected_{tc_cat}'].sum()
            ]
         for tc_cat in TC_CAT
        },
        index = ['Area affected', 'Relative area affected', 'Total area', 'Ecoregions affected']
    )
    return results

def get_vulnerability_area(nexus, ecoreg_geom_area, affected_quant=AFF_QUANT, max_rp=MAX_RP):
    df_list = []
    for cat in TC_CAT:
        df = pd.DataFrame({
            'Vulnerability' : nexus_to_affected_map(nexus[f'{cat}'], affected_quant=affected_quant).sort_values('ECO_NAME').Vulnerability.values,
            'SHAPE_AREA' : ecoreg_geom_area.sort_values('ECO_NAME').SHAPE_AREA.values
            }
        )
        df_list.append(df.groupby('Vulnerability').sum().rename(columns={'SHAPE_AREA': f'Area {cat}'}))
    return pd.concat(df_list, axis=1).applymap(round_to_e)

def get_vulnerability_ecoregions(nexus,affected_quant=AFF_QUANT):  
    vulnerability_counts = pd.DataFrame()
    for cat in TC_CAT:
        vals = nexus_to_affected_map(nexus[cat], affected_quant=affected_quant).groupby(['Vulnerability']).count()['ECO_NAME']
        vulnerability_counts[cat] = vals
    return vulnerability_counts

def get_average_rp(nexus, ecoreg_geom_area, quant='mean', max_rp=1/250):
    df = make_rp_area_table(nexus, ecoreg_geom_area, quant, max_rp)
    vals_per_cat = {}
    vulnerabilities = df['V TC1'].unique()
    for cat in TC_CAT:
        vals_per_cat[cat] = [
            df[(df[f'V {cat}'] == f'{vul}') & (df[f'Rp {cat}'] != 0)][f'Rp {cat}'].mean()
            for vul in vulnerabilities
        ]
    average_recovery_per_vulnerability = pd.DataFrame(vals_per_cat, index=vulnerabilities)
    return average_recovery_per_vulnerability


'''
Uncertainty and SEnsitivity analysis
'''

def get_output_values(ecoreg_at_risk, nexus_present, model, threshold, affected_quant):
    output = {}
    for tc_cat in TC_CAT:
        cat_name = TC_CAT_TO_NAMES[tc_cat]
        output[f'{cat_name}-intensity at risk'] = ecoreg_at_risk[threshold][f'{tc_cat}'][f'Model{model}RelDiff'].count()
        output[f'{cat_name}-intensity decrease'] = np.sum([ecoreg_at_risk[threshold][f'{tc_cat}'][f'Model{model}RelDiff'] < 0])
        output[f'{cat_name}-intensity new'] = np.sum([ecoreg_at_risk[threshold][f'{tc_cat}'][f'Model{model}RelDiff'] > 1e20])
        output[f'{cat_name}-intensity increase'] = np.sum(
            np.logical_and(
                [ecoreg_at_risk[threshold][f'{tc_cat}'][f'Model{model}RelDiff'] > 0],
                [ecoreg_at_risk[threshold][f'{tc_cat}'][f'Model{model}RelDiff'] < 1e20]
            )
        )
    for tc_cat in TC_CAT:
        cat_name = TC_CAT_TO_NAMES[tc_cat]
        nexus_cat = nexus_dict_to_affected(nexus_present, affected_quant=affected_quant)[f'{tc_cat}']
        output[f'{cat_name}-intensity dependent'] = np.sum(nexus_cat['Vulnerability'] == 'D')
        output[f'{cat_name}-intensity resilient'] = np.sum(nexus_cat['Vulnerability'] == 'R')
        output[f'{cat_name}-intensity not affected'] = np.sum(nexus_cat['Vulnerability'] == 'V')
        output[f'{cat_name}-intensity affected vulnerable'] = np.sum(nexus_cat['Vulnerability'] == 'VA')

    return output

def uncertainty_pipeline(min_prob, max_rp, model_id, res_quant, dep_quant, threshold, ecoreg_atlantic):
    res_quant = np.round(res_quant, 2)
    affected_quant = res_quant
    dep_quant = np.round(dep_quant, 2)
    threshold = np.round(threshold, 2)
    model_list = GCM + ['Median']
    model = model_list[int(model_id)]
    nexus_present = vulnerability_nexus(
        min_prob=min_prob,
        time='present',
        gcm='',
        to_file=False,
        max_rp=max_rp,
        resilience_quant=res_quant,
        dependence_quant=dep_quant
    )
    ecoreg_diff = ecoreg_diff_cc(nexus_present, ecoreg_atlantic, max_rp=max_rp, resilience_quant=res_quant, dependence_quant=dep_quant)
    ecoreg_at_risk = ecoreg_diff_at_risk(ecoreg_diff, thresholds=np.array([threshold]), model=model)
    return list(get_output_values(ecoreg_at_risk, nexus_present, model, threshold, affected_quant).values())