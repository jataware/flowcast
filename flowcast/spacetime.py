from __future__ import annotations

from enum import Enum, auto
import numpy as np
import xarray as xr
from dataclasses import dataclass
from cftime import DatetimeNoLeap
from sklearn.neighbors import BallTree

from .utilities import angle_diff



############################### TIME FUNCTIONS ################################

class Frequency(Enum):
    daily = auto()
    weekly = auto()
    monthly = auto()
    yearly = auto()
    decadal = auto()


def datetimeNoLeap_to_epoch(dt) -> float:
    """
    Convert a cftime.DatetimeNoLeap instance to an epoch timestamp.
    """
    # Define the reference datetime from which the epoch timestamp is counted.
    epoch = DatetimeNoLeap(1970, 1, 1)
    
    # Calculate the difference in seconds between the provided date and the epoch
    delta = dt - epoch
    
    # Convert the timedelta to seconds
    seconds = delta.days * 24 * 3600 + delta.seconds + delta.microseconds / 1e6

    return seconds


############################### SPACE/GEO FUNCTIONS ################################



class LongitudeConvention(Enum):
    ambiguous = auto()
    neg180_180 = auto()
    pos0_360 = auto()
    #TODO: consider adding -pi_+pi and 0_2pi
    # negπ_π = auto()
    # pos0_2π = auto()


#TODO: consider having a latitude convention
# class LatitudeConvention(Enum):
#     degrees = auto()
#     radians = auto()


@dataclass
class Resolution:
    dx: float
    dy: float = None

    def __init__(self, dx: float, dy: float|None=None):
        self.dx = dx
        self.dy = dy if dy is not None else dx


def validate_longitudes(lons:np.ndarray):
    """Validate that the given longitude values are in the range [-180, 360] and are monotonic (either positive or negative)"""
    assert np.all(lons >= -180) and np.all(lons <= 360), f'Longitude values must be in the range [-180, 360]. Got: {lons}'
    deltas = np.diff(lons)
    assert np.all(deltas >= 0) or np.all(deltas <= 0), f'Longitude values must be monotonic (either positive or negative). Got: {lons}'


def validate_latitudes(lats:np.ndarray):
    """Validate that the given latitude values are in the range [-90, 90], and are monotonic (either positive or negative)"""
    assert np.all(lats >= -90) and np.all(lats <= 90), f'Latitude values must be in the range [-90, 90]. Got: {lats}'
    deltas = np.diff(lats)
    assert np.all(deltas >= 0) or np.all(deltas <= 0), f'Latitude values must be monotonic (either positive or negative). Got: {lats}'

def determine_longitude_convention(lons:np.ndarray) -> LongitudeConvention:
    """Determine the longitude convention of the given longitude values"""

    # ensure valid longitude values
    validate_longitudes(lons)
    
    # determine the longitude convention
    if np.all(lons >= 0) and np.all(lons <= 180):
        return LongitudeConvention.ambiguous
    elif np.all(lons >= -180) and np.all(lons <= 180):
        return LongitudeConvention.neg180_180
    elif np.all(lons >= 0) and np.all(lons <= 360):
        return LongitudeConvention.pos0_360
    
    raise ValueError(f'Internal Error: Should be unreachable. Got: {lons}')
    
def convert_longitude_convention(lons:np.ndarray, target_convention:LongitudeConvention) -> np.ndarray:
    """Convert the given longitude values to the specified longitude convention"""

    assert np.all(lons >= -180) and np.all(lons <= 360), f'Longitude values must be in the range [-180, 360]. Got: {lons}'
    
    if target_convention == LongitudeConvention.ambiguous:
        target_convention = LongitudeConvention.neg180_180

    if target_convention == LongitudeConvention.neg180_180:
        return np.where(lons > 180, lons - 360, lons)
    elif target_convention == LongitudeConvention.pos0_360:
        return np.where(lons < 0, lons + 360, lons)
    else:
        raise ValueError(f'Invalid target longitude convention: {target_convention}. Expected one of: {[*LongitudeConvention.__members__.values()]}')


def inplace_set_longitude_convention(data:xr.DataArray, convention:LongitudeConvention) -> None:
    """
    Set the longitude convention of the data, and shift so that coordinates are monotonically increasing
    """
    if determine_longitude_convention(data['lon'].data) in [convention, LongitudeConvention.ambiguous]:
        return

    # map the longitude values to the new convention
    old_lon = data['lon'].data
    new_lon = convert_longitude_convention(old_lon, convention)

    # make lon the last axis for easier manipulation
    lon_idx = data.dims.index('lon')
    new_data = np.moveaxis(data.data, lon_idx, -1)

    # shift the data so that the lon coordinates are monotonically increasing
    sorted_indices = np.argsort(new_lon)
    new_lon = new_lon[sorted_indices]
    new_data = new_data[..., sorted_indices]

    # move the lon axis back, and make the data contiguous
    new_data = np.moveaxis(new_data, -1, lon_idx)
    new_data = np.ascontiguousarray(new_data)

    # update the data in place
    data.data = new_data
    data['lon'] = new_lon



def determine_tight_lon_bounds(lons: np.ndarray) -> tuple[float, float]:
    """
    Given a list of longitude coordinates in degrees, determine the tightest bounds that encompass all coordinates.
    This is useful for determining axis limits when plotting countries, etc.

    Args:
    - lons (np.ndarray): An array of longitude coordinates in degrees.

    Returns:
    - tuple: A tuple containing the left and right longitude bounds. Guarantees lon_min < lon_max, though they may not be in -180..180 range.
    """

    # compute a rough center to be a reference, and then compute all angles relative to that center
    x, y = np.cos(np.radians(lons)), np.sin(np.radians(lons))
    lon_center = np.degrees(np.arctan2(np.mean(y), np.mean(x)))
    lon_diffs = angle_diff(lons, lon_center)

    # adjust the center to be the actual center of the data
    center_adjustment = (lon_diffs.max() + lon_diffs.min()) / 2
    lon_center += center_adjustment
    lon_diffs -= center_adjustment

    # compute the bounds
    centered_lons = lon_diffs + lon_center
    lon_min, lon_max = centered_lons.min(), centered_lons.max()

    return lon_min, lon_max






def points_to_mask(lats: np.ndarray, lons: np.ndarray, /, n_lat=180, n_lon=360, min_lat=-90, max_lat=90, min_lon=-180, max_lon=180) -> xr.DataArray:
    #TODO: consider allowing for time to also be a coordinate
    """
    convert a list of points to a boolean mask that is true at each of the points

    Args:
    - lats (np.ndarray): an array of latitudes (in degrees)
    - lons (np.ndarray): an array of longitudes (in degrees)
    - n_lat (int, optional): the number of latitude bins for the output mask. Defaults to 180.
    - n_lon (int, optional): the number of longitude bins for the output mask. Defaults to 360.
    - min_lat (float, optional): the minimum latitude for the output mask. Defaults to -90.
    - max_lat (float, optional): the maximum latitude for the output mask. Defaults to 90.
    - min_lon (float, optional): the minimum longitude for the output mask. Defaults to -180.
    - max_lon (float, optional): the maximum longitude for the output mask. Defaults to 180.
    """

    # create grid
    grid_lats = np.linspace(min_lat, max_lat, n_lat+1)[:-1]
    grid_lons = np.linspace(min_lon, max_lon, n_lon+1)[:-1]

    # find the closest grid point to each point
    lat_diffs = np.abs(grid_lats - lats[:, None])
    lon_diffs = np.abs(grid_lons - lons[:, None])
    lat_idx = np.argmin(lat_diffs, axis=1)
    lon_idx = np.argmin(lon_diffs, axis=1)

    # create mask
    mask = np.zeros((n_lat, n_lon), dtype=bool)
    mask[lat_idx, lon_idx] = True

    data = xr.DataArray(mask, dims=['lat', 'lon'], coords={'lat': grid_lats, 'lon': grid_lons})

    return data

def mask_to_sdf(mask: xr.DataArray, include_initial_points:bool) -> xr.DataArray:
    """
    Generate a distance field from points in a mask

    Args:
    - mask (xr.DataArray): A boolean mask of the points to generate a distance field from. If float, non-zero values are considered True. NaNs are considered False.
    - include_initial_points (bool): Whether to include the initial points in the distance field. If False, the distance at the initial points will be NaN.
    """

    assert 'lat' in mask.coords and 'lon' in mask.coords, f'mask must have lat and lon coordinates. Got: {mask.coords.keys()}'

    # if mask only has lat/lon, it is already a single slice
    if mask.coords.keys() == {'lat', 'lon'}:
        sdf = mask_slice_to_sdf(mask.data, mask.lat, mask.lon, include_initial_points)
        return xr.DataArray(sdf, dims=['lat', 'lon'], coords={'lat': mask.lat, 'lon': mask.lon})


    #reshape mask data so that lat,lon are the last two dimensions
    mask_data = mask.data
    lat_dim_idx = mask.dims.index('lat')
    lon_dim_idx = mask.dims.index('lon')
    lon_lat_dim_idxs = [lat_dim_idx, lon_dim_idx]
    new_order = [i for i in range(len(mask.dims)) if i not in lon_lat_dim_idxs] + lon_lat_dim_idxs
    new_coords = [mask.dims[i] for i in new_order]
    mask_data = np.transpose(mask_data, axes=new_order)

    # create an empty array to store the distance field
    sdf_slices = np.zeros(mask_data.shape, dtype=float)

    # iterate over each slice of the mask
    lat, lon = mask.lat, mask.lon
    for idx in np.ndindex(mask_data.shape[:-2]):
        slice_data = mask_data[idx]
        sdf_slices[idx] = mask_slice_to_sdf(slice_data, lat, lon, include_initial_points)

    # create the new DataArray
    sdf = xr.DataArray(sdf_slices, dims=new_coords, coords={dim: mask.coords[dim] for dim in new_coords})
    return sdf



def mask_slice_to_sdf(mask: np.ndarray, lat:np.ndarray, lon:np.ndarray, include_initial_points:bool) -> np.ndarray:
    """
    Generate a distance field from points in a single (lat/lon) mask slice

    Args:
    - mask (np.ndarray): A boolean mask of the points to generate a distance field from. If float, non-zero values are considered True. NaNs are considered False.
    """

    # collect just the True the points from the mask (and convert to radians)
    assert len(mask.shape) == 2, f'mask must be 2D. Got: {mask.shape}'
    assert len(lat) == mask.shape[0] and len(lon) == mask.shape[1], f'lat and lon must have the same length as the mask. Got: {len(lat)}, {len(lon)} and {mask.shape}'

    mask_data = np.nan_to_num(mask, nan=0)
    points_x_idx, points_y_idx = np.argwhere(mask_data).T
    points_x = lat[points_x_idx]
    points_y = lon[points_y_idx]
    points = np.stack([points_x, points_y], axis=1)
    points = np.deg2rad(points)

    # create the list of points to compute the distance field for (literally just mask's coordinates)
    mesh = np.stack(np.meshgrid(lat, lon), axis=2)
    mesh_shape = mesh.shape[:2]
    mesh = np.deg2rad(mesh)
    mesh = mesh.reshape(-1, 2)

    # efficiently compute the closest point in the mask to each point in the mesh
    tree = BallTree(points, metric='haversine')
    sdf = tree.query(mesh)[0]
    sdf = sdf.reshape(*mesh_shape).T  # reshape and put lat as first dimension
    sdf *= 6371.0  # convert from radians to kilometers

    # set the distance at the initial points to NaN
    if not include_initial_points:
        sdf[points_x_idx, points_y_idx] = np.nan

    # if original mask contains NaNs, preserve them in the distance field
    if np.any(np.isnan(mask)):
        sdf[np.isnan(mask)] = np.nan

    return sdf
