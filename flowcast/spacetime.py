from __future__ import annotations

from enum import Enum, auto
import numpy as np
import xarray as xr
from dataclasses import dataclass
from cftime import DatetimeNoLeap



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

from sklearn.neighbors import BallTree
def mask_to_sdf(mask: xr.DataArray, include_initial_points:bool) -> xr.DataArray:
    """Generate a distance field from points in a boolean mask"""

    # collect just the True the points from the mask (and convert to radians)
    points_x_idx, points_y_idx = np.argwhere(mask.data).T
    points_x = mask.lat.data[points_x_idx]
    points_y = mask.lon.data[points_y_idx]
    points = np.stack([points_x, points_y], axis=1)
    points = np.deg2rad(points)

    # create the list of points to compute the distance field for (literally just mask's coordinates)
    mesh = np.stack(np.meshgrid(mask.lat.data, mask.lon.data), axis=2)
    mesh_shape = mesh.shape[:2]
    mesh = np.deg2rad(mesh)
    mesh = mesh.reshape(-1, 2)

    # efficiently compute the closest point in the mask to each point in the mesh
    tree = BallTree(points, metric='haversine')
    sdf = tree.query(mesh)[0]
    sdf = sdf.reshape(*mesh_shape).T  # reshape and put lat as first dimension
    sdf *= 6371.0  # convert from radians to kilometers

    if not include_initial_points:
        sdf[points_x_idx, points_y_idx] = np.nan

    # create DataArray from the distance field
    data = xr.DataArray(sdf, dims=['lat', 'lon'], coords={'lat': mask.lat, 'lon': mask.lon})

    return data
