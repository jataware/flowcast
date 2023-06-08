from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxr
from rioxarray.exceptions import NoDataInBounds


from regrid import regrid, get_resolution, RegridMethod


import pdb


"""
[example question]
How many people will be exposed to extreme heat events (e.g., heatwaves) in the future?
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





import xesmf as xe
def test():
    """
    How many people will be exposed to extreme heat events (e.g., heatwaves) in the future?
    """
    pop = get_population_data()
    heat = get_heatwave_data()

    # match up the years of the population data and heatwave data
    pop = pop.isel(year=slice(5,None))

    # # regrid heat% to the same resolution as population
    regridder = xe.Regridder(heat, pop, 'bilinear')
    heat = regridder(heat)


    # CDO regridding not working... doesn't support custom time unit I'm using.
    # pop_res = get_resolution(pop)
    # heat = regrid(heat, pop_res, RegridMethod.BICUBIC)
    # -- or --
    # heat_res = get_resolution(heat)
    # pop = regrid(pop, heat_res, RegridMethod.SUM)


    # threshold heat% and multiply by population
    geo_heat_exposed = ((heat > 0)['year_heat%'] * pop['population'])

    #split out the results by country
    country_heat_exposed = split_by_country(geo_heat_exposed)
    pdb.set_trace()


    # plot the results. This is the number of people per year exposed to at least one heat event (temperature > 35°C)
    heat_exposed = geo_heat_exposed.sum(dim=['lat', 'lon'])
    heat_exposed.plot()
    plt.title('People Exposed to Heatwaves by Year')
    plt.show()


import geopandas as gpd
def split_by_country(data: xr.Dataset, countries:list[str]=None) -> xr.Dataset:
    """split the data by country"""

    # ensure data has correct coordinate system
    data = data.rio.write_crs(4326)

    # load country data
    shapefile = 'gadm_0/gadm36_0.shp'
    sf = gpd.read_file(shapefile)



    # debug
    # countries = ['Ethiopia','South Sudan','Somalia','Kenya', 'Sudan', 'Eritrea', 'Somaliland', 
    #         'Djibouti','Uganda', 'Rwanda', 'Burundi', 'Tanzania']
    countries = ['China', 'India']


    # get the shapefile rows for the countries we want
    if countries is None:
        countries = sf['NAME_0'].unique().tolist()

    countries_shp = sf[sf['NAME_0'].isin(countries)]

    
    out_countries = []
    out_data = []
    
    for i, gid, country, geometry in countries_shp.itertuples():
        try:
            clipped = data.rio.clip([geometry], crs=4326)
        except NoDataInBounds:
            #TODO: handle inserting an empty array?
            continue

        # aggregate 
        country_heat = clipped.sum(dim=['lat', 'lon'])

        # save to output
        out_countries.append(country)
        out_data.append(country_heat)


    # combine all the data into one xarray
    #TODO: this isn't quite right
    out_data = xr.DataArray(out_data, dims=['country', 'year'], coords={'country': out_countries})


    pdb.set_trace()
    ...


if __name__ == '__main__':
    test()