# add parent folder so we can import data
import sys
sys.path.append('../..')

#
from data import OtherData, Scenario
import pdb
import pandas as pd
import xarray as xr
import numpy as np
from flowcast.pipeline import Pipeline, Variable, Threshold, ThresholdType
from flowcast.regrid import RegridType as GeoRegridType, RegridType as TimeRegridType
from flowcast.spacetime import points_to_mask

from matplotlib import pyplot as plt
# from mpl_toolkits.basemap import Basemap
# from scipy.spatial import cKDTree

# TODO: use sklearn.neighbors.BallTree instead
from sklearn.neighbors import BallTree




def main():
    # sdf = get_earthquake_sdf()
    quake_lats, quake_lons = get_earthquake_coords()
    quake_spots = points_to_mask(quake_lats, quake_lons, n_lat=1800, n_lon=3600)

    pipe = Pipeline()
    pipe.load('pop', OtherData.population(scenario=Scenario.ssp585))
    # pipe.load('sdf', lambda: Variable(sdf, time_regrid_type=None, geo_regrid_type=GeoRegridType.interp_or_mean))
    pipe.load('quake_spots', lambda: Variable(quake_spots, time_regrid_type=None, geo_regrid_type=GeoRegridType.interp_or_mean))
    pipe.set_geo_resolution('pop')
    pipe.set_time_resolution('pop')

    pipe.isel('pop2020', 'pop', indexers={'time':1})
    pipe.mask_to_distance_field('sdf', 'quake_spots', include_initial_points=True)
    pipe.threshold('near_quake', 'sdf', Threshold(200, ThresholdType.less_than))
    pipe.multiply('affected_pop', 'pop2020', 'near_quake')

    pipe.reverse_geocode('affected_countries', 'affected_pop', 
        places=[
            'United States',
            'Colombia',
            'Ecuador',
            'Peru',
            'Brazil',
            'Bolivia',
            'Paraguay',
            'Uruguay',
            'Argentina',
            'Chile',
            'Venezuela',
            'Guyana',
            'Suriname',
            'French Guiana',
            'China',
            'India',
            'Nepal',
            'Bhutan',
            'Bangladesh',
            'Myanmar',
            'Thailand',
            'Laos',
            'Cambodia',
            'Vietnam',
            'Malaysia',
            'Indonesia',
            'Brunei',
            'Philippines',
            'Papua New Guinea',
            'Vanuatu',
            'Taiwan',
            'Japan',
            'North Korea',
            'South Korea',
            'Mongolia',
            'Russia',
            'Kazakhstan',
            'Kyrgyzstan',
            'Tajikistan',
            'Uzbekistan',
            'Turkmenistan',
            'Afghanistan',
            'Pakistan',
            'Iran',
        ]
    )
    pipe.sum_reduce('affected_counts', 'affected_countries', dims=['lat', 'lon'])

    pipe.execute()

    # plot bar chart of top 10 countries
    res = pipe.get_last_value().data#.isel(time=1) # take 2020 time slice
    countries,values = zip(*(list(sorted(zip(res.admin0.values, res.values), key=lambda x: x[1], reverse=True))[:10]))
    plt.bar(countries, np.array(values) / 1e6)
    plt.xticks(rotation=45, ha='right')
    plt.title('People Near Earthquakes Last Month')
    plt.ylabel('Millions of People')
    plt.xlabel('Country')
    plt.show()

    # plot boolean mask of affected people
    distance_mask = pipe.get_value('near_quake').data
    distance_mask.plot()
    plt.show()

    # plot boolean mask of affected people with water masked out
    pop = pipe.get_value('pop2020').data
    distance_mask = distance_mask.astype(float)
    distance_mask.data[np.isnan(pop.data)] = np.nan
    distance_mask.plot()
    plt.show()

    pdb.set_trace()


def get_earthquake_coords():
    df = pd.read_csv('earthquakes/significant_month.csv')
    return df['latitude'].values, df['longitude'].values



def make_earthquake_mask() -> xr.Dataset:
    quake_lats, quake_lons = get_earthquake_coords()
    quake_spots = points_to_mask(quake_lats, quake_lons, n_lat=180, n_lon=360)
    pdb.set_trace()

    # convert earthquake bool dataarray to float dataset, and save
    quake_spots.data = quake_spots.data.astype(float)
    data = xr.Dataset({'epicenters': quake_spots})
    data.to_netcdf('earthquakes.nc')

    # save population data
    pop_loader = OtherData.population(scenario=Scenario.ssp585)
    pop = pop_loader().data
    data = xr.Dataset({'pop585': pop})
    data.to_netcdf('pop585.nc')


if __name__ == '__main__':
    # main()
    make_earthquake_mask()
