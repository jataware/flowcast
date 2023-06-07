import xarray as xr

from cdo import * 
import os
os.environ['HDF5_DISABLE_VERSION_CHECK'] = "1"

cdo = Cdo()
cdo.debug = False

from enum import Enum




class RegridMethod(Enum):
    SUM = ('remapsum', cdo.remapsum, 'Sum remapping, suitable for fields where the total quantity should be conserved (e.g., mass, population, water fluxes)')
    MINIMUM = ('remapmin', cdo.remapmin, 'Minimum remapping, suitable for fields where you want to preserve the minimum value within an area (e.g., minimum temperature, lowest pressure)')
    MAXIMUM = ('remapmax', cdo.remapmax, 'Maximum remapping, suitable for fields where you want to preserve the maximum value within an area (e.g., peak wind speeds, maximum temperature)')
    MEDIAN = ('remapmedian', cdo.remapmedian, 'Median remapping, suitable for fields where you want to preserve the central tendency of the data, while being less sensitive to extreme values (e.g., median income, median precipitation)')
    AVERAGE = ('remapavg', cdo.remapavg, 'Average remapping, suitable for fields representing average quantities (e.g., temperature, humidity, wind speed)')
    BILINEAR = ('remapbil', cdo.remapbil, 'Bilinear interpolation, suitable for smooth fields (e.g., temperature, pressure, geopotential height)')
    BICUBIC = ('remapbic', cdo.remapbic, 'Bicubic interpolation, suitable for smooth fields with higher-order accuracy (e.g., temperature, pressure, geopotential height)')
    CONSERVATIVE = ('remapcon', cdo.remapcon, 'First-order conservative remapping. See: https://journals.ametsoc.org/view/journals/mwre/127/9/1520-0493_1999_127_2204_fasocr_2.0.co_2.xml')
    CONSERVATIVE2 = ('remapcon2', cdo.remapcon2, 'Second-order conservative remapping. See: https://journals.ametsoc.org/view/journals/mwre/127/9/1520-0493_1999_127_2204_fasocr_2.0.co_2.xml')
    NEAREST_NEIGHBOR = ('remapnn', cdo.remapnn, 'Nearest neighbor remapping, suitable for categorical data (e.g., land use types, biome type, election area winners)')

    def __init__(self, method_name, cdo_function, description):
        self.method_name = method_name
        self.cdo_function = cdo_function
        self.description = description

    def __str__(self):
        return f'<RegridMethod.{self.name}>'
    
    def __repr__(self):
        return f'<RegridMethod.{self.name}>'


from dataclasses import dataclass

@dataclass
class Resolution:
    dx: float
    dy: float = None

    def __init__(self, dx: float, dy: float|None=None):
        self.dx = dx
        self.dy = dy if dy is not None else dx


def regrid(data: xr.Dataset, resolution: float|Resolution, method: RegridMethod) -> xr.Dataset:
    """
    Regrids the data to the target resolution using the specified aggregation method.
    """
    data.to_netcdf('tmp_data.nc')
    create_target_grid(resolution) # creates tmp_gridfile.txt

    regridded_data = method.cdo_function('tmp_gridfile.txt', input='tmp_data.nc', options='-f nc', returnXDataset=True)

    #clip the regridded data to the maximum extent of the original data
    regridded_data = regridded_data.rio.write_crs(4326)
    regridded_data = regridded_data.rio.clip_box(*data.rio.bounds())

    # Clean up temporary files
    os.remove('tmp_data.nc')
    os.remove('tmp_gridfile.txt')

    return regridded_data


def create_target_grid(resolution: float|Resolution) -> None:
    """
    Creates a target grid with the specified resolution, and saves to tmp_gridfile.txt
    """

    if not isinstance(resolution, Resolution):
        resolution = Resolution(resolution)

    # create a grid file
    content = f"""
gridtype  = latlon
xsize     = {int(360/resolution.dx)}
ysize     = {int(180/resolution.dy)}
xfirst    = {-180 + resolution.dx / 2}
xinc      = {resolution.dx}
yfirst    = {-90 + resolution.dy / 2}
yinc      = {resolution.dy}
"""
    gridfile = 'tmp_gridfile.txt'
    with open(gridfile, 'w') as f:
        f.write(content)

def multi_feature_regrid(data: xr.Dataset, resolution: float|Resolution, methods: dict[str, RegridMethod]) -> xr.Dataset:
    """
    Regrids data with multiple features using specified aggregation methods per each feature.
    """

    # collect all features that use the same aggregation method
    features_by_method = {}
    for feature, method in methods.items():
        if method not in features_by_method:
            features_by_method[method] = []
        features_by_method[method].append(feature)
    
    # regrid each group of features using the specified aggregation method    
    results = [regrid(data[features], resolution, method) for method, features in features_by_method.items()]

    # merge the results and return
    return xr.merge(results)


def get_resolution(data: xr.Dataset) -> Resolution:
    """
    Returns the resolution of the data in degrees.
    """
    dx = abs(data.lon[1] - data.lon[0]).item()
    dy = abs(data.lat[1] - data.lat[0]).item()
    return Resolution(dx, dy)