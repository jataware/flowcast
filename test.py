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

Tasks
- make time axis use correct datatype
- combine output into xarray
- test regridding back on old method

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

    # convert time integer back to datetime
    yearly_data['year'] = pd.to_datetime(yearly_data['year'].values, format='%Y')
    
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

    # convert time integer back to datetime
    heatwave['year'] = pd.to_datetime(heatwave['year'].values, format='%Y')

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
    print('collecting population and heatwave data...')
    pop = get_population_data()
    heat = get_heatwave_data()

    # match up the years of the population data and heatwave data
    pop = pop.isel(year=slice(5,None))

    # regrid heat% to the same resolution as population
    print('regridding heat data to match population data...')
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

    #plot all the countries on a single line plot
    # from flags import flag_map
    # plt.rcParams['font.family'] = 'EmojiOne Mozilla'
    for country in country_heat_exposed['country'].values:
        country_heat_exposed.sel(country=country)['heat_exposure'].plot(label=country)
        
        # # if there is a unicode flag for this country, draw the flag on the plot over the country's data
        # if country in flag_map:
        #     data = country_heat_exposed.sel(country=country)['heat_exposure']
        #     plt.text(data['time'].values[0], data[0].item(), flag_map[country], fontsize=20)



    plt.title('People Exposed to Heatwaves by Country')
    plt.legend()
    plt.show()


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


    # DEBUG
    # countries = ['Ethiopia','South Sudan','Somalia','Kenya', 'Sudan', 'Eritrea', 'Somaliland', 'Djibouti','Uganda', 'Rwanda', 'Burundi', 'Tanzania']
    # pdb.set_trace()
    # countries = ['China', 'India', 'United States', 'Canada', 'Mexico']

    # if no countries are specified, use all of them
    if countries is None:
        countries = sf['NAME_0'].unique().tolist()

    # get the shapefile rows for the countries we want
    countries_set = set(countries)
    countries_shp = sf[sf['NAME_0'].isin(countries_set)]

    # sort the countries in countries_shp to match the order of the countries in the data
    countries_shp = countries_shp.set_index('NAME_0').loc[countries].reset_index()

    
    out_data = np.zeros((len(data['year']), len(countries)))
    
    for i, (_, gid, country, geometry) in enumerate(countries_shp.itertuples()):
        print(f'processing {country}...')
        try:
            clipped = data.rio.clip([geometry], crs=4326)
        except NoDataInBounds:
            # countries with no overlap will get the default of all zeros
            continue

        # aggregate for this country
        country_heat = clipped.sum(dim=['lat', 'lon'])
        del clipped

        # save to output
        out_data[:, i] = country_heat

    # combine all the data into one xarray
    out_data = xr.Dataset(
        {
            'heat_exposure': (['time', 'country'],  out_data)
        },
        coords={
            'time': data['year'].values,
            'country': countries
        }
    )

    return out_data




def test2():
    import xarray as xr
    import pandas as pd
    import numpy as np

    # First, let's create some example data
    countries = ['USA', 'Canada', 'Mexico']
    years = pd.date_range('2000-01-01', periods=20, freq='Y')
    temperature_data = np.random.rand(len(years), len(countries))

    # Next, we create a Dataset
    ds = xr.Dataset(
        {
            'temperature': (['time', 'country'],  temperature_data)
        },
        coords={
            'time': years,
            'country': countries
        }
    )

    print(ds)

    pdb.set_trace()
    ...





if __name__ == '__main__':
    test()
    # test2()