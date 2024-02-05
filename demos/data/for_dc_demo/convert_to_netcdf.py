import numpy as np
import pandas as pd
from cftime import DatetimeNoLeap
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
from pyproj import Transformer
from typing import Literal
import xarray as xr
from pathlib import Path
import os

import pdb




def create_netcdf(dataset: Literal['asset_wealth', 'consumption']) -> xr.DataArray:
    # Open the TIFF file
    with rasterio.open(f'november_tests_atlasai_{dataset.replace("_", "")}_allyears_2km.tif') as src:
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
    geo_df = gpd.read_file(f'november_tests_{dataset}.geojson')
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




if __name__ == '__main__':
    #move to the directory with the data
    cwd = Path.cwd()
    origin = Path(__file__).parent.absolute()
    os.chdir(origin.parent / 'asset_wealth'/'AtlasAI')

    #create dataarrays
    consumption = create_netcdf('consumption')
    asset_wealth = create_netcdf('asset_wealth')

    #merge dataarrays
    data = xr.merge([consumption, asset_wealth])

    #take the last frame
    data = data.isel(date=-1)
    data = data.drop('date')

    #save to netcdf to the original directory
    data.to_netcdf(Path(origin / 'test_wealth.nc'))
    os.chdir(cwd)