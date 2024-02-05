from flowcast.pipeline import Pipeline, Variable, Threshold, ThresholdType
from flowcast.regrid import RegridType as TimeRegridType, RegridType as GeoRegridType
import xarray as xr
from matplotlib import pyplot as plt
from itertools import count
import numpy as np

import numpy as np

##
import pdb


ree_indicators = {
    "Geology_Eon_Maximum_Majority": ["Precambrian"],
    "Geology_Eon_Maximum_Minority": ["Precambrian"],
    "Geology_Eon_Minimum_Majority": ["Precambrian"],
    "Geology_Eon_Minimum_Minority": ["Precambrian"],

    "Geology_Era_Maximum_Majority": ["Proterozoic"],
    "Geology_Era_Maximum_Minority": ["Proterozoic"],
    "Geology_Era_Minimum_Majority": ["Proterozoic"],
    "Geology_Era_Minimum_Minority": ["Proterozoic"],

    "Geology_Period_Maximum_Majority": ["Neoarchean", "Paleoproterozoic", "Mesoproterozoic"],  # lots
    "Geology_Period_Maximum_Minority": ["Neoarchean", "Paleoproterozoic", "Mesoproterozoic"],
    "Geology_Period_Minimum_Majority": ["Neoarchean", "Paleoproterozoic", "Mesoproterozoic"],
    "Geology_Period_Minimum_Minority": ["Neoarchean", "Paleoproterozoic", "Mesoproterozoic"],

    "Geology_Lithology_Majority": ["Igneous_Intrusive_Alkalic", "Metamorphic_Amphibolite"],  # little
    "Geology_Lithology_Minority": ["Igneous_Intrusive_Alkalic", "Metamorphic_Amphibolite"],

    "Geology_Dictionary_Alkalic": ["Present"],  # lots
    # # Numeric data thresholds will need expert input for specific values
    # "Geology_PassiveMargin_Proximity": "numeric_threshold",  # Replace with actual threshold
    # "Geology_Fault_Proximity": "numeric_threshold",         # Replace with actual threshold
    # "Crust1_CrustalThickness": "numeric_threshold",         # Replace with actual threshold
    # "Crust1_SedimentThickness": "numeric_threshold",        # Replace with actual threshold
    # # Similarly, seismic, gravity, and magnetic data need thresholds or ranges
    # "Seismic_LAB_Hoggard": "numeric_threshold",             # Replace with actual threshold
    # "Seismic_LAB_Priestley": "numeric_threshold",           # Replace with actual threshold
    # "Seismic_Moho": "numeric_threshold",                    # Replace with actual threshold
    # "Gravity_GOCE_Differential": "numeric_threshold",       # Replace with actual threshold
    # # ... (include other relevant numeric features with their thresholds)
    "Crust1_Type": ["continental arc", "island arc", "orogen, thick upper crust, fast middle crust", "orogen (Antarctica), thick upper crust, thin lower crust"]
}


def main():
    # Geology data
    data = xr.open_dataset('au_datacube.nc')

    # MODIS data
    modis = xr.open_rasterio('AU_MODIS/DLCD_v2-1_MODIS_EVI_13_20140101-20151231.tif')
    modis = modis.isel(band=0, drop=True)
    modis = modis.rename({'x': 'lon', 'y': 'lat'})
    modis.data = modis.data.astype(np.float32)


    def loader_factory(feature):
        return lambda: Variable(feature, None, geo_regrid_type=GeoRegridType.nearest_or_mode)

    pipe = Pipeline()
    pipe.set_geo_resolution('Geology_Era_Maximum_Majority')
    pipe.set_time_resolution('Geology_Era_Maximum_Majority')

    # load all the geology data via the loaders
    for key in ree_indicators:
        pipe.load(key, loader_factory(data[key]))

    # load the modis data
    pipe.load('modis', lambda: Variable(modis, time_regrid_type=TimeRegridType.nearest_or_mode, geo_regrid_type=GeoRegridType.nearest_or_mode))

    # create a distance field from modis
    pipe.threshold('water_mask', 'modis', Threshold(3, ThresholdType.equal))
    pipe.mask_to_distance_field('sdf', 'water_mask', include_initial_points=False)

    # create masks for each feature
    threshold_names = []
    for key, thresholds in ree_indicators.items():
        for threshold in thresholds:
            legend: list[str] = data[key].legend
            threshold_value = legend.index(threshold)
            threshold_name = f'{key}_{threshold}_threshold'
            pipe.threshold(threshold_name, key, Threshold(threshold_value, ThresholdType.equal))
            threshold_names.append(threshold_name)

    # combine masks
    pipe.add('acc0', threshold_names[0], threshold_names[1])
    for i, threshold_name in zip(count(1), threshold_names[2:]):
        pipe.add(f'acc{i}', f'acc{i-1}', threshold_name)

    # divide by the sdf
    pipe.scalar_add('sdf_plus_1', 'sdf', 1)
    pipe.divide('final', f'acc{i}', 'sdf_plus_1')

    # execute the pipeline
    pipe.execute()

    # plot the result (mark anywhere the value was max)
    last = pipe.get_last_value().data

    # (optional) replace ocean with nan
    # nanmask = np.isnan(pipe.get_value('Geology_Era_Maximum_Majority').data.data)
    # last.data[nanmask] = np.nan

    (np.log(last + 0.01)-np.log(0.01)).plot()

    # collect and plot the top 10 points
    scores = last.data.flatten()
    scores = scores[~np.isnan(scores)]
    scores = np.sort(scores)
    threshold = scores[-10]
    best_ys, best_xs = np.where((last.data >= threshold))
    plt.scatter(last.lon[best_xs], last.lat[best_ys], s=100, c='red', marker='x')

    plt.show()

    # for key, thresholds in ree_indicators.items():
    #     for threshold in thresholds:
    #         threshold_name = f'{key}_{threshold}_threshold'
    #         pipe.get_value(threshold_name).data.plot()
    #         plt.show()

    pdb.set_trace()
    ...


# def get_neighbors(mask: np.ndarray):
#     """
#     return a boolean mask of neighbors relative to the original mask

#     neighbors only include 4-connected neighbors (up, down, left, right)
#     """
#     # structuring_element = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
#     structuring_element = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
#     return binary_dilation(mask, structure=structuring_element) & ~mask


# def make_distance_atlas(lats: np.ndarray, lons: np.ndarray):
#     r_earth = 6371.0  # kilometers

#     # convert to radians
#     lats = np.radians(lats)
#     lons = np.radians(lons)

#     dlats = np.diff(lats)
#     dlat = dlats[0]
#     assert np.allclose(dlats, dlat), 'latitudes must be evenly spaced'

#     dlons = np.diff(lons)
#     dlon = dlons[0]
#     assert np.allclose(dlons, dlon), 'longitudes must be evenly spaced'

#     step_sizes_x = dlon * r_earth * np.cos(lats)
#     step_sizes_y = dlat * r_earth * np.ones_like(lats)

#     return step_sizes_x, step_sizes_y


# def haversine(lats: np.ndarray, lons: np.ndarray, y: np.ndarray, x: np.ndarray):
#     """
#     return the distance from each point in the first set of coordinates every point in the second set of coordinates

#     Args:
#         lats: 1D array of latitudes (in degrees)
#         lons: 1D array of longitudes (in degrees)
#         y: 1D array of latitudes (in degrees)
#         x: 1D array of longitudes (in degrees)

#     returns:
#         matrix of distances for each pair (in kilometers)
#     """
#     # set up for broadcasting
#     lats = lats[:, None]
#     lons = lons[:, None]
#     y = y[None]
#     x = x[None]

#     r_earth = 6371.0  # kilometers

#     # convert to radians
#     lats = np.radians(lats)
#     lons = np.radians(lons)
#     y = np.radians(y)
#     x = np.radians(x)

#     # calculate the distance
#     dlat = lats - y
#     dlon = lons - x
#     a = np.sin(dlat/2)**2 + np.cos(y) * np.cos(lats) * np.sin(dlon/2)**2
#     c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
#     d = r_earth * c

#     return d


# def generate_AU_sdf():
#     data = xr.open_rasterio('AU_MODIS/DLCD_v2-1_MODIS_EVI_13_20140101-20151231.tif')
#     ###### DEBUG reshape the data with flowcast ############
#     # Necessary because the current algorithm is pretty inefficient
#     print('regridding data...', end='', flush=True)
#     data.data = data.data.astype(np.float32)
#     old_x = data.x.data
#     old_y = data.y.data
#     new_x = np.arange(old_x.min(), old_x.max(), 0.1)
#     new_y = np.arange(old_y.min(), old_y.max(), 0.1)
#     data = regrid_1d(data, new_x, 'x', aggregation=GeoRegridType.nearest_or_mode)
#     data = regrid_1d(data, new_y, 'y', aggregation=GeoRegridType.nearest_or_mode)
#     print('done')

#     ###### DEBUG take smaller selection of data ############
#     # data = data.isel(y=slice(8950, 9550), x=slice(10800, 11200))

#     data = data.isel(band=0)
#     data = data.rename({'x': 'lon', 'y': 'lat'})
#     raw_data = data.data
#     lats = data.lat.data
#     lons = data.lon.data

#     # create blank float32 array
#     sdf = np.full(data.shape, np.nan, dtype=np.float32)

#     # mask to select all bodies of water in the data
#     init_mask = raw_data == 3

#     mask_ys, mask_xs = np.where(init_mask)
#     mask_lats = lats[mask_ys]
#     mask_lons = lons[mask_xs]

#     chunk_size = 5
#     for coords in tqdm(chunked(product(range(lats.size), range(lons.size)), chunk_size), total=(lats.size * lons.size)//chunk_size, desc='calculating distances'):
#         ys, xs = np.array(coords).T
#         distances = haversine(mask_lats, mask_lons, lats[ys], lons[xs])
#         # bests = distances.argmin(axis=0)
#         # sdf[ys, xs] = distances[bests]
#         sdf[ys, xs] = distances.min(axis=0)

#     # save the SDF
#     sdf = xr.DataArray(sdf, coords=[data.lat, data.lon], dims=['lat', 'lon'])
#     sdf.to_netcdf('sdf.nc')


if __name__ == '__main__':
    # generate_AU_sdf()
    main()
