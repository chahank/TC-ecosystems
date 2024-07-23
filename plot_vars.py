"""
Created 2024

@author: Chahan M. Kropf
"""

import cartopy

import geopandas as gpd

from pathlib import Path
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from helper_methods import DATA, storm_ecoreg_geoms_in_centroids
import seaborn as sns

import cartopy.crs as ccrs

ECO_REG_GEOM = Path('./Data/') / Path(storm_ecoreg_geoms_in_centroids(time='present', gcm='')).with_suffix('.shz')

CCRS_ATLANTIC = ccrs.EckertIV(central_longitude=0)
PROJ_ATLANTIC = CCRS_ATLANTIC.proj4_init

CCRS_PACIFIC = ccrs.EckertIV(central_longitude=160)
PROJ_PACIFIC = CCRS_PACIFIC.proj4_init

LAT_BOUNDS = (-60, 60)
LAT_BOUNDS_PROJ = (-6e6, 7.25e6)
LAT_TROPICS = (-40, 40)
LAT_TROPICS_PROJ = (-5e6, 5e6)
LON_BOUNDS = (-180, 180)
LON_BOUNDS_PROJ = (-2e7, 2e7)
LON_BOUNDS_RISK = (-1.7e7, 1.53e7)

CMAP_AFFECTED = LinearSegmentedColormap.from_list('yr', ['Lightgrey', "indianred"], N=2) 
CMAP_VULNERABILITIES = LinearSegmentedColormap.from_list('yr', ["darkcyan", "chocolate", 'Lightgrey'], N=3) 
CMAP_VULNERABILITIES_AFFECTED = LinearSegmentedColormap.from_list('yr', ["darkcyan", "chocolate", 'lightsteelblue', 'Lightgrey'], N=4) #['Dependent', 'Not Affected', 'Resilient', 'Vulnerable']
CMAP_POS = sns.color_palette("rocket_r", as_cmap=True)
CMAP_NEG = LinearSegmentedColormap.from_list('yr',['darkcyan', 'aquamarine'], N=256) 
CMAP_AGREEMENT = LinearSegmentedColormap.from_list('yr', ["forestgreen", "gray"], N=2)
NEW_AFFECTED_COLOR = 'mediumpurple'

OCEAN_COLOR = 'azure'

TC_CAT_TO_NAMES = {
    'TC1': 'Low',
    'TC2': 'Middle',
    'TC3': 'High'
}

def oceans_plot(centered_on='atlantic'):
    oceans = [x for x in cartopy.feature.OCEAN.geometries()]
    oceans_gdf = gpd.GeoDataFrame([1, 2], geometry=oceans).set_crs(4326)
    if centered_on == 'atlantic':
        return oceans_gdf.set_crs(4326).to_crs(PROJ_ATLANTIC)
    elif centered_on == 'pacific':
        return oceans_gdf.set_crs(4326).to_crs(PROJ_PACIFIC)

def ecoreg_plot(centered_on='atlantic'):
    eco_reg_geom = gpd.GeoDataFrame.from_file(ECO_REG_GEOM).set_crs(4326)
    eco_reg_geom = eco_reg_geom.sort_values('ECO_NAME')
    if centered_on == 'atlantic':
        return eco_reg_geom.set_crs(4326).to_crs(PROJ_ATLANTIC)
    elif centered_on == 'pacific':    
        return eco_reg_geom.set_crs(4326).to_crs(PROJ_PACIFIC)





