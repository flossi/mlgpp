#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 13:59:46 2022

@author: hermanns
"""

import pandas as pd
pd.set_option('display.max_columns', 6)
pd.set_option('display.max_rows', 20)
from pathlib import Path
import datetime as dt

import fiona
fiona.supported_drivers['KML'] = 'rw'

from fmch.hsicos import HSICOS, _build_icos_meta

'''
Folder structure:
Input flux data (e.g. CSV files from ICOS ETC L2 Archives)
must be put in /data/fluxes. Input hyperspectral (PRISMA) data must be put in
/data/PRISMA. The data base of PRISMA imagery used in the analysis is located
in /data/prisma_db.csv.

Output files are generated in /out. The folder already contains output .gpkgs
that are ready for use in statistical analysis with prisma_gpp_stats_main.R.

Python modules are stored in /fmch/fmch and R functions are stored in /R.

'''

#%% Load flux data
flx = _build_icos_meta(save=True) # saving GPKG is required for stat. analysis in R

wdirexp = Path(__file__)

dateparse = lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
flx_prisma = pd.read_csv(wdirexp / 'data' / 'prisma_db.csv', parse_dates=['startdate'],
                        date_parser=dateparse, dtype={'dataTakeID': str}).reset_index(drop=True)
flx_prisma['date'] = [dt.datetime.date(x).isoformat() for x in flx_prisma['startdate']]
flx_prisma['icostime'] = (flx_prisma.startdate + pd.Timedelta(hours=1)).dt.round('30min').dt.time

#%% Combining flux and hyperspectral data with HSICOS class

icos_list0 = flx_prisma.name.unique().tolist()
'''Setting a custom working directory is possible but the subfolder structure
of this repository should be preserved.'''
db_name = 'prisma_db.csv'
prisma_gpp = HSICOS(img_csv=db_name, do_mkdir=True, out_dir='hsicos_test')

# Import geometry from ICOS L2 data and target (projected) CRS info from imagery
prisma_gpp.crs_and_cropping(icos_list0, zip_path=prisma_gpp.img_dir, overwrite=True, save_csv=True)

# QC overviews
test_qcdf = prisma_gpp.hsi_qc(icos_site=icos_list0, save=True)
'''After manual inspection of imagery, add a 'usable' column to the image .csv
(here: prisma_db.csv) and enter for each image: 0 for unusable, 1 for usable,
2 for maybe usable (requires closer inspection of target geometries).'''

# Check the ICOS data base: Which sites need ERA5 PBLH data?
era_blh_list = prisma_gpp.var_check(icos_list0)

# Download PBLH values from ERA5 & precip + temp from E-OBS via Copernicus CDS
prisma_gpp.icos_cds_get(era_blh_list, ts=True)
prisma_gpp.icos_cds_get(icos_list0, eobs=True)
# Calculate SPEI components from E-OBS data
prisma_gpp.icos_eobs_pet(icos_list0)

# SPEI computation in R: prisma_gpp_spei_main.R

'''Exlude unusable imagery and integrate ERA5 PBLH values for stations that do
not record PBLH (but all other flux footprint (FF) variables)'''
icos_list = prisma_gpp.update_img_db(db_name, era_blh=era_blh_list, save=True)

# Cropping of HSI (+ FF modeling if required)
# 1. VNIR, FF, reflectance
prisma_ff_ref_vnir, image_cubes = prisma_gpp.hsi_geom_crop(
    icos_list, sr='vnir', save=True)
# 2. VNIR, buffer geometries, reflectance
prisma_bg_ref_vnir, _ = prisma_gpp.hsi_geom_crop(
    icos_list, sr='vnir', zonal=True, save=True)
# 3. VIS, FF, upwelling radiation (UPW)
prisma_ff_upw_vis, _ = prisma_gpp.hsi_geom_crop(
    icos_list, sr='vis', upw=True, save=True)
# 4. VIS, FF, reflectance
prisma_ff_ref_vis, _ = prisma_gpp.hsi_geom_crop(
    icos_list, sr='vis', save=True)
# 5. VSWIR, FF, reflectance
prisma_ffp_ref_swir, _ = prisma_gpp.hsi_geom_crop(
    icos_list, sr='vswir', save=True)

# Reloading data (without covariates) examples
prisma_ffp_ref = prisma_gpp.load_cropped_db()
prisma_zon_ref = prisma_gpp.load_cropped_db(sr='vnir', zonal=True)

# Download Copernicus PPI data
prisma_gpp.icos_ppi_get(icos_list, dataset = 'VI', day_range=10)

# Add SPEI & PPI values to data frame (adapt to other processing configs)
prisma_gpp.hsi_add_spei_ppi(zonal=False, save=True)

# Reloading full data examples
prisma_gpp_ref = prisma_gpp.load_spei_ppi_db()
prisma_gpp_ref_cov_sum = prisma_gpp.load_spei_ppi_db(sr='vnir', zonal=True)
