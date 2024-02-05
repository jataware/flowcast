from __future__ import annotations

#add ../.. to the path so we can import from the parent directory
import sys; sys.path.append('../..')

import numpy as np
from flowcast.pipeline import Pipeline, Frequency
from data import OtherData, CMIP6Data, Realization, Scenario, Model
from flowcast.spacetime import DatetimeNoLeap, LongitudeConvention, inplace_set_longitude_convention
from flowcast.regrid import regrid_1d, RegridType
from matplotlib import pyplot as plt
import xarray as xr
import pandas as pd
import rasterio
import geopandas as gpd
from pyproj import Transformer
from typing import Literal
from pathlib import Path

import pdb


def crop_to_data(data:xr.DataArray, crop_value=None) -> xr.DataArray:
    """Crop the dataset to fit tightly around all non-nan values"""
    #find the first and last non-nan values in each dimension
    mask = np.isnan(data.values)
    if crop_value is not None:
        mask |= (data.values == crop_value)
    mask = ~mask

    indexers = {}
    for i, dim in enumerate(data.dims):
        #shift the current dimension to be first in the mask
        shifted_mask = np.moveaxis(mask, i, 0)
        slices = shifted_mask.reshape(len(data[dim]), -1)
        dim_mask = np.any(slices, axis=1)
        first = np.argmax(dim_mask)
        last = len(dim_mask) - np.argmax(dim_mask[::-1])
        indexers[dim] = slice(first, last)
    return data.isel(indexers)

def make_pr():
    pipe = Pipeline(low_memory=True)
    # pipe.set_time_resolution(Frequency.yearly)

    pipe.load('pr', CMIP6Data.pr(realization=Realization.r1i1p1f1, scenario=Scenario.ssp585, model=Model.FGOALS_f3_L))
    pipe.reverse_geocode('pr2','pr', ['Djibouti', 'Somalia', 'Kenya', 'Uganda', 'South Sudan', 'Sudan', 'Ethiopia', 'Yemen'])
    pipe.sum_reduce('pr3', 'pr2', ['admin0'])

    pipe.execute()
    pr = pipe.get_value('pr3').data

    #ensure the lon conventions match
    inplace_set_longitude_convention(pr, LongitudeConvention.neg180_180)

    #convert from dataarray to dataset and save
    pr = crop_to_data(pr, crop_value=0.0)
    pr = pr.isel(time=slice(1,None)) #chop off the first frame
    pr = pr.to_dataset(name='pr')

    plt.figure()
    pr.isel(time=0)['pr'].plot()
    plt.show()

    pr.to_netcdf('pr.nc')

    #DEBUG get original unmodified data
    pr=pipe.get_value('pr').data


    pdb.set_trace()

def make_pop():

    # pipeline for population data
    pipe = Pipeline(low_memory=True)
    pipe.load('pop', OtherData.population(scenario=Scenario.ssp585))
    pipe.reverse_geocode('pop2','pop', ['Djibouti', 'Somalia', 'Kenya', 'Uganda', 'South Sudan', 'Sudan', 'Ethiopia', 'Yemen'])
    pipe.sum_reduce('pop3', 'pop2', ['admin0'])

    pipe.execute()
    pop = pipe.get_value('pop3').data

    #ensure the lon conventions match
    inplace_set_longitude_convention(pop, LongitudeConvention.neg180_180)

    pop = crop_to_data(pop, crop_value=0.0)
    pop = pop.to_dataset(name='population')

    plt.figure()
    np.log(pop.isel(time=0)['population'] + 1e-6).plot()
    plt.show()

    #convert to 32 bit float
    pop.to_netcdf('pop.nc')



    pdb.set_trace()


def make_modis():

    pipe = Pipeline(low_memory=True)
    pipe.load('modis', OtherData.land_cover())
    pipe.reverse_geocode('modis2','modis', ['Djibouti', 'Somalia', 'Kenya', 'Uganda', 'South Sudan', 'Sudan', 'Ethiopia', 'Yemen'])
    pipe.sum_reduce('modis3', 'modis2', ['admin0'])

    pipe.execute()
    modis = pipe.get_value('modis3').data

    #ensure the lon conventions match
    inplace_set_longitude_convention(modis, LongitudeConvention.neg180_180)

    modis = crop_to_data(modis, crop_value=0.0)
    modis = modis.isel(time=-1).drop('time')
    modis = modis.to_dataset(name='land_cover')

    plt.figure()
    modis['land_cover'].plot()
    plt.show()

    modis.to_netcdf('modis.nc')

    pdb.set_trace()




def create_asset_wealth_netcdf(root:Path, dataset: Literal['asset_wealth', 'consumption']) -> xr.DataArray:
    # Open the TIFF file
    with rasterio.open(root / f'november_tests_atlasai_{dataset.replace("_", "")}_allyears_2km.tif') as src:
        data = src.read()
        transform = src.transform
        crs = src.crs

    num_bands, height, width = data.shape
    rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    x_coords, y_coords = transform * (cols, rows)


    # If the data is not in a geographic coordinate system, convert to one (e.g., EPSG:4326)
    if not crs.is_geographic:
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        longitudes, latitudes = transformer.transform(x_coords, y_coords)
    else:
        longitudes, latitudes = x_coords, y_coords

    # check that the latitudes/longitudes are square
    assert np.all(np.diff(latitudes, axis=1) == 0), 'Latitudes must be rectangular'
    assert np.all(np.diff(longitudes, axis=0) == 0), 'Longitudes must be rectangular'


    # pull time dimension from geojson
    geo_df = gpd.read_file(root / f'november_tests_{dataset}.geojson')
    times = geo_df['model_prediction_start_date'].unique()
    pandas_dates = pd.to_datetime(times)
    cftime_dates = np.array([DatetimeNoLeap(date.year, date.month, date.day) for date in pandas_dates])


    # Create an xarray DataArray
    da = xr.DataArray(
        data,
        dims=["date", "latitude", "longitude"],
        coords={
            "date": cftime_dates,
            "latitude": latitudes[:, 0],  # Assuming regular grid, take the first column
            "longitude": longitudes[0, :]  # Assuming regular grid, take the first row
        },
        name=dataset
    )


    return da




def make_asset():
    root = Path(__file__).absolute().parent.parent / 'asset_wealth' / 'AtlasAI'
    #create dataarrays
    consumption = create_asset_wealth_netcdf(root, 'consumption')
    asset_wealth = create_asset_wealth_netcdf(root, 'asset_wealth')

    #merge dataarrays
    data = xr.merge([consumption, asset_wealth])

    #take the last frame
    data = data.isel(date=-1)
    data = data.drop('date')

    #plot
    plt.figure()
    data['consumption'].plot()
    plt.show()

    #save to netcdf to the original directory
    data.to_netcdf('wealth.nc')




















if __name__ == '__main__':
    make_pr()
    # make_pop()
    # make_modis()
    # make_asset()