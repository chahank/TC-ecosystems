"""
Created 2024

@author: Chahan M. Kropf
"""

import copy

import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd 
import pandas as pd

from pathlib import Path
from scipy import sparse

from climada.engine import ImpactCalc
from climada.hazard import Hazard
from climada.entity import Exposures

DATA = Path("./Data/")
STORM_FOLDER = DATA / Path('./STORM/Global')

SAFFIR_CAT = {'TC': np.array([ 17.5,  33,  49,  58,  200])}
TC_CAT = ['TC1', 'TC2', 'TC3']
GCM = ['CMCC-CM2-VHR4', 'CNRM-CM6-1-HR', 'EC-Earth3P-HR', 'HadGEM3-GC31-HM']

RES = '150as'
SCE = 'rcp85'
ECO_REG = DATA / Path("./Ecoregions2017/Ecoregions2017.shp")

'''
Make frequency hazards
'''

def storm_tc_filename(time, gcm, resolution=RES):
    if time == 'present':
        return f'present/TC_STORM_{resolution}_global.hdf5'
    elif time =='future':
        return f'future/TC_STORM_{gcm}_{resolution}_global.hdf5'
    return None

def storm_freqs_filename(time, gcm, resolution=RES, scenario=SCE):
    if time == 'present':
        return f'freqs_tc_storm_{resolution}'
    if time == 'future':
        return f'freqs_tc_storm_{resolution}_{gcm}_{scenario}'

def storm_ecoreg_freqs_filename(time, gcm, resolution=RES, scenario=SCE):
    if time == 'present':
        return f'freqs_ecoreg_tc_storm_{resolution}'
    if time == 'future':
        return f'freqs_ecoreg_tc_storm_{resolution}_{gcm}_{scenario}'

def storm_ecoreg_freqs_quantile_filename(tc_cat, time, gcm, resolution=RES, scenario=SCE):
    if time == 'present':
        return f'freqs_quantiles_ecoreg_tc_storm_{resolution}_{tc_cat}'
    if time == 'future':
        return f'freqs_quantiles_ecoreg_tc_storm_{resolution}_{gcm}_{scenario}_{tc_cat}'
    
def storm_ecoreg_geoms_in_centroids(time, gcm, resolution=RES, scenario=SCE):
    if time == 'present':
        return f'ecoreg_geoms_tc_storm_{resolution}'
    if time == 'future':
        return f'ecoreg_geoms_tc_storm_{resolution}_{gcm}_{scenario}'

def haz_to_cathaz(haz, cat):
    haz_cat = copy.deepcopy(haz)
    haz_cat.intensity.data = np.digitize(haz.intensity.data, cat) - 1
    haz_cat.intensity.eliminate_zeros()
    return haz_cat

def _make_freq_gdf(freqs_dict, centroids):
    centroids.set_geometry_points()
    gdf_freqs = gpd.GeoDataFrame(freqs_dict, geometry=centroids.geometry.values, crs='EPSG:4326')
    return gdf_freqs

def freqs_per_cat_gdf(haz_cat):
    cat = np.unique(haz_cat.intensity.data).astype(int)
    haz_type = haz_cat.haz_type
    haz_freqs_dict = {}
    for c in cat:
        cat_freq = ImpactCalc.eai_exp_from_mat((haz_cat.intensity == c).astype(int), haz_cat.frequency)
        cat_freq = np.squeeze(np.asarray(cat_freq))
        haz_freqs_dict[haz_type + str(c)] = cat_freq
    return _make_freq_gdf(haz_freqs_dict, haz_cat.centroids)

def load_tc_STORM(time, gcm, resolution=RES, folder=STORM_FOLDER):
    return Hazard.from_hdf5(Path(folder) / storm_tc_filename(time, gcm, resolution))

def make_tc_freqs_storm(time, gcm, resolution=RES, scenario=SCE, cat=SAFFIR_CAT, folder=STORM_FOLDER, to_file=False):
    tc = load_tc_STORM(gcm=gcm, time=time, resolution=resolution, folder=folder)
    haz_cat = haz_to_cathaz(tc, cat[tc.haz_type])
    freqs_df = freqs_per_cat_gdf(haz_cat)
    if to_file:
        print('Save tc frequency to file')
        freqs_df.to_file(DATA / Path(storm_freqs_filename(time, gcm, resolution, scenario)).with_suffix('.shz'))
    return freqs_df

def make_storm_freqs_ecoreg(time, gcm, resolution=RES, scenario=SCE, cat=SAFFIR_CAT, tc_folder=STORM_FOLDER, to_file=False):
    print('Make or read tc frequency')
    tc_freq_file_path = DATA / Path(storm_freqs_filename(time, gcm, resolution, scenario)).with_suffix('.shz')
    if tc_freq_file_path.is_file():
        freqs_df = gpd.read_file(tc_freq_file_path)
    else:
        freqs_df = make_tc_freqs_storm(time, gcm, resolution=resolution, scenario=scenario, cat=cat, folder=tc_folder, to_file=to_file)
    print('Read ecoregion')
    eco_reg_gdf = gpd.read_file(ECO_REG)
    print('Join tc and ecoregion')
    freqs_eco_reg_gdf = freqs_df.sjoin(eco_reg_gdf, how='inner')
    if to_file:
        print('Save ecoregion frequency to file')
        freqs_eco_reg_gdf.to_file(DATA / Path(storm_ecoreg_freqs_filename(time, gcm, resolution=resolution, scenario=scenario)).with_suffix('.shz'))
    return freqs_eco_reg_gdf

def make_one_cat_freqs_quantiles(freqs_gdf, geo_names_column, tc_cat, quantiles, mean=True, std=True, filename=''):
    print(f'Compute frequency quantiles for {tc_cat}')
    freqs_quant = freqs_gdf.groupby(geo_names_column)[tc_cat].quantile(quantiles, interpolation='nearest')
    freqs_quant = freqs_quant.unstack()
    freqs_quant.columns = np.around(freqs_quant.columns, 2)
    if mean:
        m = freqs_gdf.groupby(geo_names_column)[tc_cat].mean()
        freqs_quant.insert(0, 'mean', m)
    if std:
        s = freqs_gdf.groupby(geo_names_column)[tc_cat].std()
        freqs_quant.insert(0, 'std', s)
    if len(str(filename)) > 0 :
        print('Save quantiles to csv')
        freqs_quant.to_csv(Path(filename).with_suffix(f'.csv'))
    return freqs_quant

def make_storm_freqs_quantile(time, gcm, resolution=RES, scenario=SCE, cat=SAFFIR_CAT, tc_folder=STORM_FOLDER, to_file=False):
    freqs_file_path = DATA / Path(storm_ecoreg_freqs_filename(time, gcm, resolution=resolution, scenario=scenario)).with_suffix('.shz')
    if freqs_file_path.is_file():
        print('Read ecoreg freqs file')
        ecoreg_freqs_gdf = gpd.read_file(freqs_file_path)
    else:
        ecoreg_freqs_gdf = make_storm_freqs_ecoreg(
            time, gcm, resolution=resolution, scenario=scenario, cat=cat, tc_folder=tc_folder, to_file=to_file
        )  
    geo_names_column = 'ECO_NAME'
    quantiles = np.linspace(0, 1, 101)
    mean=True
    std=True
    quantiles_df_list = []
    for tc_cat in TC_CAT:
        if to_file:
            filename = DATA / storm_ecoreg_freqs_quantile_filename(tc_cat, time, gcm, resolution, scenario)
        else:
            filename = ''
        if Path(filename).is_file():
            print(f'Read csv for quantiles {tc_cat}')
            quantiles_df_list.append(pd.read_csv(Path(filename).with_suffix('.csv')))
        else:
            quantiles_df_list.append(make_one_cat_freqs_quantiles(ecoreg_freqs_gdf, geo_names_column, tc_cat, quantiles, mean=mean, std=std, filename=filename))
    return quantiles

def make_ecoreg_geoms_in_centroids(time, gcm, resolution=RES, scenario=SCE, cat=SAFFIR_CAT, tc_folder=STORM_FOLDER, to_file=False):
    freqs_file_path = DATA / Path(storm_ecoreg_freqs_filename(time, gcm, resolution=resolution, scenario=scenario)).with_suffix('.shz')
    if freqs_file_path.is_file():
        print('Read ecoreg freqs file')
        ecoreg_freqs_gdf = gpd.read_file(freqs_file_path)
    else:
        ecoreg_freqs_gdf = make_storm_freqs_ecoreg(
            time, gcm, resolution=resolution, scenario=scenario, cat=cat, tc_folder=tc_folder, to_file=to_file
        )  
    selected_ecoregs = ecoreg_freqs_gdf.ECO_NAME.unique()
    eco_reg_gdf = gpd.read_file(ECO_REG)
    eco_geoms = eco_reg_gdf[['OBJECTID', 'ECO_NAME', 'BIOME_NAME', 'REALM', 'ECO_ID', 'SHAPE_AREA','geometry']][np.isin(eco_reg_gdf.ECO_NAME.values, selected_ecoregs)]
    if to_file:
        filename = DATA / Path(storm_ecoreg_geoms_in_centroids(time, gcm, resolution, scenario)).with_suffix('.shz')
        eco_geoms.to_file(filename)
    return eco_geoms