from __future__ import annotations

from enum import Enum, auto
import numpy as np
import xarray as xr
from dataclasses import dataclass
from cftime import DatetimeNoLeap



############################### TIME FUNCTIONS ################################

class Frequency(Enum):
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
