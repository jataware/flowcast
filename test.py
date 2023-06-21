from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxr # needs to be imported?
from rioxarray.exceptions import NoDataInBounds


from regrid import regrid, get_resolution, RegridMethod


import pdb


"""
[example question]
How many people will be exposed to extreme heat events (e.g., heatwaves) in the future?

Tasks
- make heat scenario allow selection of SSP scenario
- make heat scenario regrid before thresholding

"""


from enum import Enum

class Scenario(str, Enum):
    SSP126 = 'ssp126'
    SSP245 = 'ssp245'
    SSP370 = 'ssp370'
    SSP585 = 'ssp585'



def get_population_data(ssp:Scenario) -> xr.Dataset:
    """get an xarray with the specified population data"""
    
    ssp = ssp.value[:-2] # remove the last two characters (e.g., 'ssp126' -> 'ssp1')

    years = [*range(2010, 2110, 10)]
    all_data = [xr.open_dataset(f'data/population/{ssp.upper()}/Total/NetCDF/{ssp}_{year}.nc') for year in years]

    for i, year in enumerate(years):
        data = all_data[i]
        # rename the population variable to be consistent
        data = data.rename({f'{ssp}_{year}': 'population'})#, 'lon': 'x', 'lat': 'y'})

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




def get_heatwave_data(ssp:Scenario) -> xr.Dataset:
    
    # get cmip6 data for maximum temperature according to the given ssp scenario
    data = xr.open_dataset(f'data/cmip6/tasmax/tasmax_Amon_CAS-ESM2-0_{ssp}_r1i1p1f1_gn_201501-210012.nc')
    
    #threshold data to only tasmax values that are larger than 35°C (308.15 K)
    mask = data['tasmax'] > 308.15
    mask['year'] = data['time.year']
    heatwave = mask.groupby('year').mean(dim='time')

    # convert data array to dataset, and rename the variable
    heatwave = heatwave.to_dataset().rename({'tasmax': 'year_heat%'})

    # convert time integer back to datetime
    heatwave['year'] = pd.to_datetime(heatwave['year'].values, format='%Y')

    return heatwave





import xesmf as xe
def extreme_heat_scenario(ssp:Scenario):
    """
    How many people will be exposed to extreme heat events (e.g., heatwaves) in the future?
    """
   
    print('collecting population and heatwave data...')
    pop = get_population_data(ssp)
    heat = get_heatwave_data(ssp)

    # match up the years of the population data and heatwave data
    pop = pop.isel(year=slice(5,None))

    # regrid heat% to the same resolution as population
    print('regridding heat data to match population data...')
    regridder = xe.Regridder(heat, pop, 'bilinear')
    heat = regridder(heat)


    # CDO regridding not working... result seems to be empty after multiplying by population
    # print('regridding heat data to match population data...')
    # pop_res = get_resolution(pop)
    # heat = regrid(heat, pop_res, RegridMethod.BILINEAR)
    # -- or --
    # heat_res = get_resolution(heat)
    # pop = regrid(pop, heat_res, RegridMethod.SUM)


    # threshold heat% and multiply by population
    geo_heat_exposed = ((heat > 0)['year_heat%'] * pop['population'])

    #split out the results by country
    country_heat_exposed = split_by_country(geo_heat_exposed, countries=['China', 'India', 'United States', 'Canada', 'Mexico'])

    # plot all the countries on a single plot
    for country in country_heat_exposed['country'].values:
        country_heat_exposed.sel(country=country)['heat_exposure'].plot(label=country)

    plt.title('People Exposed to Heatwaves by Country')
    plt.legend()
    plt.show()

    return country_heat_exposed



import geopandas as gpd
def split_by_country(data: xr.Dataset, countries:list[str]=None) -> xr.Dataset:
    """split the data by country"""

    # ensure data has correct coordinate system
    data = data.rio.write_crs(4326)

    # load country data
    shapefile = 'gadm_0/gadm36_0.shp'
    sf = gpd.read_file(shapefile)


    # if no countries are specified, use all of them
    if countries is None:
        countries = sf['NAME_0'].unique().tolist()

    # get the shapefile rows for the countries we want
    countries_set = set(countries)
    countries_shp = sf[sf['NAME_0'].isin(countries_set)]

    # sort the countries in countries_shp to match the order of the countries in the data
    countries_shp = countries_shp.set_index('NAME_0').loc[countries].reset_index()

    
    out_data = np.zeros((len(data['year']), len(countries)))
    
    for i, (_, country, gid, geometry) in enumerate(countries_shp.itertuples()):
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



from cftime import DatetimeNoLeap

def crop_viability_scenario(ssp:Scenario, viability_threshold:float=3.0) -> xr.Dataset:
    # load modis data and select cropland layer
    modis = xr.open_dataset('data/MODIS/land-use-5km.nc')
    modis = modis['LC_Type1']

    # convert modis time from Julian (🤣) to match the pr and tas data
    modis['time'] = modis.indexes['time'].to_datetimeindex().map(lambda dt: DatetimeNoLeap(dt.year, dt.month, dt.day))

    # load pr and tas data with chunking
    chunk_size = {'time': 1}  # Adjust chunk size based on your memory availability and the size of your dataset
    pr = xr.open_dataset(f'data/cmip6/pr/pr_Amon_FGOALS-f3-L_{ssp}_r1i1p1f1_gr_201501-210012.nc', chunks=chunk_size)
    tas = xr.open_dataset(f'data/cmip6/tas/tas_Amon_FGOALS-f3-L_{ssp}_r1i1p1f1_gr_201501-210012.nc', chunks=chunk_size)

    # regrid pr and tas to match modis
    regridder = xe.Regridder(pr, modis, 'bilinear')
    pr = regridder(pr)
    tas = regridder(tas)

    # handle interpolation artifact:
    tas = tas.where(tas != 0, np.nan)
    pr = pr.where(pr != 0, np.nan)

    # select the time period from tas/pr that is within the range of time in modis
    tas_slice = tas.sel(time=slice(modis['time'].min(), modis['time'].max()))
    pr_slice = pr.sel(time=slice(modis['time'].min(), modis['time'].max()))

    # Take the first frame of the modis data as the location of cropland
    # TODO: look into some sort of mode or other aggregation over all the modis data
    cropland = modis.isel(time=0)
    cropmask = cropland == 12


    # take mean/std of tas/pr over all cropland. this is the baseline to compare against
    tas_mean = tas_slice.where(cropmask).mean(dim=['lat', 'lon', 'time'], skipna=True)['tas'].values.item()
    tas_std = tas_slice.where(cropmask).std(dim=['lat', 'lon', 'time'], skipna=True)['tas'].values.item()
    pr_mean = pr_slice.where(cropmask).mean(dim=['lat', 'lon', 'time'], skipna=True)['pr'].values.item()
    pr_std = pr_slice.where(cropmask).std(dim=['lat', 'lon', 'time'], skipna=True)['pr'].values.item()


    # calculate the z-scores for the full pr and tas datasets (only considering cropland)
    tas_z = ((tas - tas_mean) / tas_std).where(cropmask)
    pr_z = ((pr - pr_mean) / pr_std).where(cropmask)

    # take |z| < 3 as the threshold for viability
    # total viability is where both pr and tas are suitable
    viability_mask = ((np.abs(tas_z) < viability_threshold)['tas'] & (np.abs(pr_z) < viability_threshold)['pr'])
    viability = viability_mask.sum(dim=['lat', 'lon']) / cropmask.sum(dim=['lat', 'lon'])

    # aggregate each year's viability into a single value
    viability = viability.groupby('time.year').mean(dim='time')

    # convert to xarray.dataset
    viability = xr.Dataset(
        {
            'viability': (['year'], viability.values)
        },
        coords={
            'year': viability['year'].values
        }
    )

    # plot the viability over time
    viability['viability'].plot()
    plt.title(f'Crop viability over time ({ssp})')
    plt.xlabel('Year')
    plt.ylabel('Suitability')
    plt.show()

    return viability







if __name__ == '__main__':
    import sys

    ssps = [*Scenario.__members__.keys()]
    if len(sys.argv) != 2:
        print(f'usage: python test.py <{"|".join(ssps)}>')
        sys.exit(1)

    try:
        ssp = Scenario(sys.argv[1])
    except ValueError:
        raise ValueError(f'invalid scenario. expected one of: {", ".join(ssps)}. got: {sys.argv[1]}') from None

    # run a scenario
    extreme_heat_scenario(ssp)
    # crop_viability_scenario(ssp)