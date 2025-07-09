import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from tensorflow import keras
import xarray as xr
import cartopy
import matplotlib.animation as animation
from datetime import datetime, timedelta
import cartopy.crs as ccrs
import cartopy.feature
from collections import Counter
from scipy.signal import detrend

from scipy.stats import ttest_ind
import statistics
import seaborn as sns
import teleconnection_data_format as telecon
from statsmodels.stats.multitest import fdrcorrection
import cartopy.feature as cfeature

act = xr.open_dataset('../pper_hail_1979_2019.nc')
preds_hail = xr.open_dataset('../files/models_wout_tele6_1.nc')
era_data = xr.open_dataset('../files/era_cin_cape_data.nc') 
olr_data = xr.open_dataset('../files/olr_avec_min15.nc')


# actual hail formating 
act_hail = xr.open_dataset('../pper_hail_1979_2019.nc')
act_arr_hail = np.zeros(preds_hail.hail.shape)
tmp_act = act.sel(time = preds_hail.time.values, method="nearest")

for tt, day in enumerate(tmp_act.time.values):
    tslice = slice(day, day + np.timedelta64(6, 'D'))
    act_arr_hail[tt, :, :, :93] = act_hail.sel(time = tslice).p_perfect_hail.values[:, :64, :]

### Setup for ERA5 Variables

act = xr.open_dataset('../pper_hail_1979_2019.nc')
preds = xr.open_dataset('../files/models_wout_tele6_1.nc')
era_data = xr.open_dataset('../files/era_cin_cape_data.nc') 


# take out seasonality -> 21-day running mean for each year



# take out effects of climate change on data
tmp_dtr = detrend(era_data['t800'], axis=0)
u250_dtr = detrend(era_data['u250'], axis=0)
z500_dtr = detrend(era_data['z500'], axis=0)
z50_dtr = detrend(era_data['z50'], axis=0)


lead_times = list(range(7,14))
comp_days = [7, 14, 21, 28]

# placing variables back into era_data
era_data['t800'] = (('time', 'lon', 'lat'), tmp_dtr)
era_data['u250'] = (('time', 'lon', 'lat'), u250_dtr)
era_data['z500'] = (('time', 'lon', 'lat'), z500_dtr)
era_data['z50'] = (('time', 'lon', 'lat'), z50_dtr)

yrs = list(range(1983, 2019))
for yr in yrs:
    tm_slicer = slice(str(yr)+'-11-01', str(yr+1)+'-05-31')
    smoothed = era_data.sel(time = tm_slicer).rolling(time = 21, center=True).mean()
    smoothed = smoothed.fillna(era_data.sel(time = tm_slicer))

    era_data.loc[dict(time=tm_slicer)] = smoothed

era_data = era_data.assign_coords(lon=(era_data.lon % 360)).sortby('lon')



# for lead_time in lead_times:
lead_time = 7
var = 'u250'

plt.rcParams['hatch.linewidth'] = 1.2 


for lead_time in lead_times:

    # Define color levels
    p_levels = np.linspace(2, 10, 8)
    n_levels = np.linspace(-10, -2, 8)

    era_ttst = era_data.sel(lon=np.arange(100, 302.5, 2.5), lat=np.arange(17.5, 92.5, 2.5), method='nearest')

    # Create a 2x2 grid of subplots with PlateCarree projection
    fig, axs = plt.subplots(2, 2, figsize=(16, 10), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})

    for idx, comp_day in enumerate(comp_days):
        row = idx // 2
        col = idx % 2
        ax = axs[row, col]

        top_ten_pcnt_tims = find_top_ten_percent_of_days(lead_time=lead_time)
        var_tmp_full = np.zeros((len(top_ten_pcnt_tims), 81, 30))

        for tt, time in enumerate(top_ten_pcnt_tims):
            var_tmp_full[tt, :, :] = era_data[var].sel(
                lon=np.arange(100, 302.5, 2.5), 
                lat=np.arange(17.5, 92.5, 2.5),
                time=time - np.timedelta64(comp_day, 'D'),
                method='nearest'
            )

        print(var_tmp_full.shape)
        print(era_ttst[var].shape)

        t_stat, t_pval = ttest_ind(var_tmp_full, era_ttst[var], axis=0, equal_var=False)

        pvals_flat = t_pval.flatten()
        rej, pval_corr = fdrcorrection(pvals_flat, alpha=0.01)
        sig_mask = rej.reshape(t_pval.shape).astype(float)
        sig_mask[sig_mask == 0] = np.nan


        # Get static grid for plotting (any valid time works)
        north_pacific_ds = era_data.sel(
            lon=np.arange(100, 302.5, 2.5),
            lat=np.arange(17.5, 92.5, 2.5),
            time='2018-05-01', method='nearest'
        )

        ax.add_feature(cartopy.feature.LAND, facecolor='white')
        ax.add_feature(cartopy.feature.OCEAN, facecolor='white', alpha=0.5)
        ax.add_feature(cartopy.feature.COASTLINE, linewidth=0.5)
        ax.add_feature(cartopy.feature.LAKES, alpha=0.5)
        ax.add_feature(cartopy.feature.STATES, linewidth=0.5)
        ax.set_extent([100, 300, 17.5, 87.5], crs=ccrs.PlateCarree())


    
        data = t_stat.T


        contours_pos = ax.contourf(
            north_pacific_ds.lon,
            north_pacific_ds.lat,
            data,
            levels=p_levels,
            cmap='YlOrRd',
            transform=ccrs.PlateCarree()
        )

        contours_neg = ax.contourf(
            north_pacific_ds.lon,
            north_pacific_ds.lat,
            data,
            levels=n_levels,
            cmap='Blues_r',
            transform=ccrs.PlateCarree()
        )

        ax.contour(
            north_pacific_ds.lon,
            north_pacific_ds.lat,
            data,
            levels=p_levels,
            colors='black',
            linewidths=1,
            transform=ccrs.PlateCarree()
        )

        ax.contour(
            north_pacific_ds.lon,
            north_pacific_ds.lat,
            data,
            levels=n_levels,
            colors='black',
            linestyles='--',
            linewidths=1,
            transform=ccrs.PlateCarree()
        )

        ax.contourf(
            north_pacific_ds.lon,
            north_pacific_ds.lat,
            sig_mask.T,  # transpose to match lat-lon orientation
            hatches=['.....'],
            colors='none',
            transform=ccrs.PlateCarree()
        )

        ax.set_title(f'Composite {comp_day} Days Prior')

    # Add a single colorbar for all subplots
    # Colorbar for positive contours
    cax_pos = fig.add_axes([0.55, 0.08, 0.25, 0.02])
    cb_pos = fig.colorbar(contours_pos, cax=cax_pos, orientation='horizontal')
    cb_pos.set_label(f'{var} Positive Anomalies')

    # Colorbar for negative contours
    cax_neg = fig.add_axes([0.25, 0.08, 0.25, 0.02])
    cb_neg = fig.colorbar(contours_neg, cax=cax_neg, orientation='horizontal')
    cb_neg.set_label(f'{var} Negative Anomalies')

    plt.suptitle(f'{var.upper()} Composite Maps for the Top 10% of Day-'+str(lead_time)+' Predictions', fontsize=20, y=0.9)
    plt.tight_layout(rect=[0, 0.1, 1, 0.93])  # Leave space for supertitle and colorbar
    plt.show()


