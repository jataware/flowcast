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

# def get_earthquake_sdf():
#     df = pd.read_csv('earthquakes/significant_month.csv')



#     # collect the columns 'latitude', 'longitude', 'mag' and convert to an xarray dataarray
#     df = df[['latitude', 'longitude', 'mag']]
#     df = df.rename(columns={'latitude': 'lat', 'longitude': 'lon', 'mag': 'magnitude'})

#     #convert to boolean mask for events
#     # mask = event_to_mask(df.lat.values, df.lon.values, df.time.values)
#     mask = event_to_mask(df.lat.values, df.lon.values)
#     sdf = mask_to_sdf(mask)
#     # pdb.set_trace()


#     # # convert points to distance field
#     # sdf = event_to_sdf2(df.lat.values, df.lon.values)

#     return sdf

# def event_to_mask(lats: np.ndarray, lons: np.ndarray, /, n_lat=180, n_lon=360, min_lat=-90, max_lat=90, min_lon=-180, max_lon=180) -> xr.DataArray:
#     """
#     convert events latitudes and longitudes to a boolean mask
#     """

#     grid_lats = np.linspace(min_lat, max_lat, n_lat+1)
#     grid_lons = np.linspace(min_lon, max_lon, n_lon+1)

#     lat_diffs = np.abs(grid_lats - lats[:, None])
#     lon_diffs = np.abs(grid_lons - lons[:, None])

#     lat_idx = np.argmin(lat_diffs, axis=1)
#     lon_idx = np.argmin(lon_diffs, axis=1)

#     # create mask
#     mask = np.zeros((n_lat, n_lon), dtype=bool)
#     mask[lat_idx, lon_idx] = True

#     data = xr.DataArray(mask, dims=['lat', 'lon'], coords={'lat': grid_lats, 'lon': grid_lons})

#     return data
    

# def mask_to_sdf(mask: xr.DataArray):
#     points_x_idx, points_y_idx = np.argwhere(mask.data).T
#     points_x = mask.lat.data[points_x_idx]
#     points_y = mask.lon.data[points_y_idx]
#     points = np.stack([points_x, points_y], axis=1)
#     points = np.deg2rad(points)

#     mesh = np.stack(np.meshgrid(mask.lat.data, mask.lon.data), axis=2)
#     mesh_shape = mesh.shape[:2]
#     mesh = np.deg2rad(mesh)
#     mesh = mesh.reshape(-1, 2)

#     tree = BallTree(points, metric='haversine')

#     sdf = tree.query(mesh)[0]
#     sdf = sdf.reshape(*mesh_shape).T  # reshape and put lat as first dimension
#     sdf *= 6371.0  # convert from radians to kilometers

#     data = xr.DataArray(sdf, dims=['lat', 'lon'], coords={'lat': mask.lat, 'lon': mask.lon})


#     return data




    # p_lats_rad = np.deg2rad(p_lats_deg)
    # p_lons_rad = np.deg2rad(p_lons_deg)
    # points = np.stack([p_lats_rad, p_lons_rad], axis=1)

    # grid_n_lat = 2500
    # grid_n_lon = 5000

    # lats_deg = np.linspace(-90, 90, grid_n_lat)
    # lats_rad = np.deg2rad(lats_deg)
    # lons_deg = np.linspace(-180, 180, grid_n_lon)
    # lons_rad = np.deg2rad(lons_deg)

    # grid_lats, grid_lons = np.meshgrid(lats_rad, lons_rad)
    grid = np.stack([grid_lats, grid_lons], axis=2)
    grid = grid.reshape(-1, 2)

    tree = BallTree(points, metric='haversine')

    sdf = tree.query(grid)[0]
    sdf = sdf.reshape(*grid_lats.shape).T  # reshape and put lat as first dimension
    sdf *= 6371.0  # convert from radians to kilometers

    data = xr.DataArray(sdf, dims=['lat', 'lon'], coords={'lat': lats_deg, 'lon': lons_deg})

    return data




def lat_lon_to_xyz(lats: np.ndarray, lons: np.ndarray):
    """
    convert latitudes and longitudes to xyz coordinates on a sphere
    """
    # convert to radians
    lats = np.radians(lats)
    lons = np.radians(lons)

    # convert to cartesian coordinates
    x = np.cos(lats) * np.cos(lons)
    y = np.cos(lats) * np.sin(lons)
    z = np.sin(lats)

    return x, y, z


# def event_to_sdf(lats: np.ndarray, lons: np.ndarray):
#     x, y, z = lat_lon_to_xyz(lats, lons)
#     points = np.stack([x, y, z], axis=1)

#     n_lat = 5000
#     n_lon = 5000
#     lats = np.linspace(-90, 90, n_lat)
#     lons = np.linspace(-180, 180, n_lon)

#     grid_lon, grid_lat = np.meshgrid(lons, lats)
#     grid_x, grid_y, grid_z = lat_lon_to_xyz(grid_lat, grid_lon)
#     grid = np.stack([grid_x, grid_y, grid_z], axis=2)

#     # Use cKDTree for efficient nearest neighbor search
#     tree = cKDTree(points)

#     # query the tree for the distance to the nearest neighbor
#     sdf = tree.query(grid)[0]

#     plt.imshow(sdf)
#     plt.show()

#     pdb.set_trace()


# TODO:allow adjusting bounds & number of points of sdf
def event_to_sdf2(p_lats_deg: np.ndarray, p_lons_deg: np.ndarray) -> xr.DataArray:
    # x, y, z = lat_lon_to_xyz(lats, lons)
    p_lats_rad = np.deg2rad(p_lats_deg)
    p_lons_rad = np.deg2rad(p_lons_deg)
    points = np.stack([p_lats_rad, p_lons_rad], axis=1)

    grid_n_lat = 2500
    grid_n_lon = 5000

    lats_deg = np.linspace(-90, 90, grid_n_lat)
    lats_rad = np.deg2rad(lats_deg)
    lons_deg = np.linspace(-180, 180, grid_n_lon)
    lons_rad = np.deg2rad(lons_deg)

    grid_lats, grid_lons = np.meshgrid(lats_rad, lons_rad)
    grid = np.stack([grid_lats, grid_lons], axis=2)
    grid = grid.reshape(-1, 2)

    tree = BallTree(points, metric='haversine')

    sdf = tree.query(grid)[0]
    sdf = sdf.reshape(*grid_lats.shape).T  # reshape and put lat as first dimension
    sdf *= 6371.0  # convert from radians to kilometers

    data = xr.DataArray(sdf, dims=['lat', 'lon'], coords={'lat': lats_deg, 'lon': lons_deg})

    return data


if __name__ == '__main__':
    main()
