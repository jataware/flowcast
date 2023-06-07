from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from regrid import regrid, get_resolution, RegridMethod


import pdb


"""
[example question]
How many people will be exposed to extreme heat events (e.g., heatwaves) in the future?

Tasks:
- define what counts as an extreme heat event
- get projected population data
- combine the two and aggregate
- final plot should be a line graph of predicted people exposed (per year)
"""

def get_population_data() -> xr.Dataset:
    """get an xarray with SSP5 population data"""
    
    years = [*range(2010, 2110, 10)]
    data_folder = 'data/population/SSP5/Total/NetCDF'

    all_data = [xr.open_dataset(f'{data_folder}/ssp5_{year}.nc') for year in years]

    for i, year in enumerate(years):
        data = all_data[i]
        # rename the population variable to be consistent
        data = data.rename({f'ssp5_{year}': 'population'})#, 'lon': 'x', 'lat': 'y'})

        # add a year coordinate
        data['decade'] = year #pd.Timestamp(year, 1, 1)

        # reassign back to the list of data
        all_data[i] = data

    # combine all the data into one xarray
    all_data = xr.concat(all_data, dim='decade')

    return all_data


def test():
    """
    How many people will be exposed to extreme heat events (e.g., heatwaves) in the future?
    """
    pop = get_population_data()
    heat = get_heatwave_data()

    pop_res = get_resolution(pop)
    heat_res = get_resolution(heat)

    # # regrid heat% to the same resolution as population
    # heat = regrid(heat, pop_res, RegridMethod.BICUBIC)

    # regrid population to the same resolution as heat%
    pop = regrid(pop, heat_res, RegridMethod.SUM)

    # multiply heat% by population
    heat_exposed = heat * pop

    pdb.set_trace()
    ...




def get_heatwave_data() -> xr.Dataset:
    
    # get cmip6 data and population data
    datapath = 'data/cmip6/tasmax_Amon_CanESM5_ssp585_r13i1p2f1_gn_201501-210012.nc'
    data = xr.open_dataset(datapath)
    

    #threshold data to only tasmax values that are larger than 35°C (308.15 K)
    mask = data['tasmax'] > 308.15
    mask['decade'] = data['time.year'] // 10 * 10
    heatwave = mask.groupby('decade').mean(dim='time') * 100

    # # TODO: figure out how to rename the variable...
    # percentage_by_decade = percentage_by_decade.rename({'tasmax': 'pct-heatwave'})

    # # plot all 10 decades in a 2x5 plot
    # fig, axes = plt.subplots(2, 5, figsize=(20, 10))
    # for i, ax in enumerate(axes.flat):
    #     decade = 2000 + (i+1) * 10
    #     heatwave.isel(decade=i).plot(ax=ax, vmin=0, vmax=100)
    #     ax.set_title(f'{decade}s')

    # #set the title for the entire plot
    # fig.suptitle('Extreme Heat Event Percentage by Decade', fontsize=20)
    # plt.show()

    ...

    return heatwave



if __name__ == '__main__':
    test()