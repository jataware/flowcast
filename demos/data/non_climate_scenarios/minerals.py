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

    # plot all of the masks
    # for key, thresholds in ree_indicators.items():
    #     for threshold in thresholds:
    #         threshold_name = f'{key}_{threshold}_threshold'
    #         pipe.get_value(threshold_name).data.plot()
    #         plt.show()

    pdb.set_trace()
    ...



if __name__ == '__main__':
    main()
