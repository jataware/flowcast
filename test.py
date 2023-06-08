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

    # Interpolate to yearly resolution
    yearly_data = all_data.interp(decade=np.arange(2010, 2101))
    yearly_data = yearly_data.rename({'decade': 'year'})
    
    return yearly_data

import xesmf as xe
def test():
    """
    How many people will be exposed to extreme heat events (e.g., heatwaves) in the future?
    """
    pop = get_population_data()
    heat = get_heatwave_data()

    # match up the years of the population data and heatwave data
    pop = pop.isel(year=slice(5,None))

    # regrid heat% to the same resolution as population
    regridder = xe.Regridder(heat, pop, 'bilinear')
    heat = regridder(heat)


    # CDO regridding not working...
    # pop_res = get_resolution(pop)
    # heat_res = get_resolution(heat)
    # heat = regrid(heat, pop_res, RegridMethod.BICUBIC)
    # pop = regrid(pop, heat_res, RegridMethod.SUM)

    # threshold heat% and multiply by population
    heat_exposed = ((heat > 0)['year_heat%'] * pop['population']).sum(dim=['lat', 'lon'])

    # plot the results. This is the number of people per year exposed to at least one heat event (temperature > 35°C)
    heat_exposed.plot()
    plt.title('People Exposed to Heatwaves by Year')
    plt.show()




def get_heatwave_data() -> xr.Dataset:
    
    # get cmip6 data and population data
    datapath = 'data/cmip6/tasmax_Amon_CanESM5_ssp585_r13i1p2f1_gn_201501-210012.nc'
    data = xr.open_dataset(datapath)
    

    #threshold data to only tasmax values that are larger than 35°C (308.15 K)
    mask = data['tasmax'] > 308.15
    mask['year'] = data['time.year']
    heatwave = mask.groupby('year').mean(dim='time')

    # convert data array to dataset, and rename the variable
    heatwave = heatwave.to_dataset().rename({'tasmax': 'year_heat%'})

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