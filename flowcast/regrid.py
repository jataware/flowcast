from __future__ import annotations

from enum import Enum, auto
import numpy as np
from scipy import stats
import xarray as xr
from warnings import warn
from .spacetime import datetimeNoLeap_to_epoch



class BinOffset(Enum):
    left = auto()
    # center = auto()  #TODO: for now don't use this
    # right = auto()   #TODO: for now don't use this

class RegridType(Enum):
    conserve = auto()
    min = auto()
    max = auto()
    mean = auto()    #TODO: consider removing in favor of just interp_or_mean
    median = auto()
    mode = auto()    #TODO: consider removing in favor of just nearest_or_mode
    interp_or_mean = auto()
    nearest_or_mode = auto()

# which methods weight the values of the bins on reduction
float_weighted_reduction_methods: set[RegridType] = {
    RegridType.mean,
    RegridType.interp_or_mean,
    RegridType.conserve,
}

# which methods take all values of bind unmodified on reduction 
boolean_reduction_methods: set[RegridType] = {
    RegridType.min,
    RegridType.max,
    RegridType.median,
    RegridType.mode,
    RegridType.nearest_or_mode,
}

assert all(method in float_weighted_reduction_methods or method in boolean_reduction_methods for method in RegridType), 'Some regrid methods are not accounted for in the float_weighted_reduction_methods or boolean_reduction_methods sets'


def compute_overlap(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Given two arrays of bin edges, compute the amount of overlap between each bin in a and each bin in b

    Parameters:
        a (np.ndarray): The first array of bin edges
        b (np.ndarray): The second array of bin edges

    Returns:
        np.ndarray: A matrix of shape (len(a), len(b)) where each element is the proportion overlap between the corresponding bins in a and b
    """
    # Guarantee that a and b are in ascending order
    if (a_reversed := a[1] < a[0]):
        a = a[::-1]
    if (b_reversed := b[1] < b[0]):
        b = b[::-1]

    # Calculate overlap using ascending logic since they're both in the same order now
    a = a[:, None]
    overlap = np.maximum(0, np.minimum(a[1:], b[1:]) - np.maximum(a[:-1], b[:-1]))
    
    # Reverse the overlap matrix back so indices match the original arrays
    if a_reversed:
        overlap = overlap[::-1]
    if b_reversed:
        overlap = overlap[:, ::-1]
    
    return np.ascontiguousarray(overlap, dtype=np.float64)


def is_resolution_increase(overlaps:np.ndarray, inclusive:bool=False) -> bool:
    """
    Determine whether the resolution is increasing or decreasing based on the overlaps matrix
    """
    nonzero_indices = np.nonzero(overlaps)
    old_size = nonzero_indices[0].max() - nonzero_indices[0].min()
    new_size = nonzero_indices[1].max() - nonzero_indices[1].min()

    if inclusive:
        return new_size >= old_size
    else:
        return new_size > old_size

def get_interp_or_mean_overlaps(overlaps:np.ndarray, old_coords:np.ndarray, new_coords:np.ndarray) -> np.ndarray:
    """
    Convert the overlaps matrix to one that will perform:
    - interpolation for resolution increases
    - mean aggregation for resolution decreases
    """

    # resolution decrease just uses existing overlaps matrix
    if not is_resolution_increase(overlaps):
        return overlaps

    # compute the distances from each old cell to each new cell
    distances = np.abs(old_coords[:, None] - new_coords[None, :])

    #zero out everything except for the closest two cells in each column
    distance_ranks = np.argsort(distances, axis=0)
    distances[distance_ranks[2:], np.arange(distances.shape[1])] = np.nan

    #invert the distances so the closest cell has the largest weight
    distances = np.nansum(distances, axis=0)[None] - distances

    # normalize so that each column sums up to 1
    distances_sum = np.nansum(distances, axis=0)
    distances[:, distances_sum > 0] /= distances_sum[distances_sum > 0]

    # replace nans with 0
    distances[np.isnan(distances)] = 0

    return distances

def get_nearest_or_mode_overlaps(overlaps:np.ndarray, old_coords:np.ndarray, new_coords:np.ndarray) -> np.ndarray:
    """
    Get an overlaps matrix that just selects the item nearest to the location of the new bin
    - nearest neighbor for resolution increases
    - mode aggregation for resolution decreases
    """
    
    # resolution decrease just uses existing overlaps matrix
    if not is_resolution_increase(overlaps):
        return overlaps

    # compute the distances from each old cell to each new cell
    distances = np.abs(old_coords[:, None] - new_coords[None, :])

    #zero out everything except for the closest cell in each column, and set the closest cell to 1
    distance_ranks = np.argsort(distances, axis=0)
    distances[distance_ranks[1:], np.arange(distances.shape[1])] = 0
    distances[distances > 0] = 1

    return distances


def get_bins(coords:np.ndarray, offset:BinOffset) -> np.ndarray:
    """
    Given an array of evenly spaced coordinates, return the bin edges

    Parameters:
        coords (np.ndarray): The array of evenly spaced coordinates
        offset (BinOffset): Whether the bins should be left, centered, or right of the coordinates

    Returns:
        np.ndarray, np.ndarray: The bin edges, and the deltas between each bin
    """
    if len(coords) < 2:
        raise ValueError('Cannot infer bin edges from a single coordinate')
        
    deltas = coords[1:] - coords[:-1]
    if not np.allclose(deltas, deltas[0]):
        #This happens for e.g. monthly timestamps where months can have different numbers of days
        warn(f'coords are not evenly spaced. Got spacings: {np.unique(deltas)}', RuntimeWarning)

    if offset == BinOffset.left:
        bins = np.concatenate([[coords[0] - deltas[0]], coords])
    elif offset == BinOffset.center:
        bins = np.concatenate([[coords[0] - deltas[0]], coords]) + np.concatenate([[deltas[0]], deltas, [deltas[-1]]]) / 2
    elif offset == BinOffset.right:
        bins = np.concatenate([coords, [coords[-1] + deltas[-1]]])

    # update the deltas to include the edges
    deltas = np.diff(bins)

    return bins, deltas



def regrid_1d(
        data:xr.DataArray,
        new_coords:np.ndarray,
        dim:str,
        offset:BinOffset=BinOffset.left,
        aggregation=RegridType.interp_or_mean,
        wrap:tuple[float,float]=None, #TBD format for this...
        low_memory:bool=False,
    ) -> xr.DataArray:
    
    # grab the old coords and data (copy data so we don't modify the original)
    old_coords = data[dim].data
    old_data = data.data.copy()

    # convert time coords to epoch timestamps
    if dim == 'time':
        old_coords = np.array([datetimeNoLeap_to_epoch(time) for time in old_coords])
        new_coords_copy = new_coords.copy()
        new_coords = np.array([datetimeNoLeap_to_epoch(time) for time in new_coords])

    # get the bin boundaries for the old and new data
    old_bins, old_deltas = get_bins(old_coords, offset)
    new_bins, _ = get_bins(new_coords, offset)
    
    #compute the amount of overlap between each old bin and each new bin
    overlaps = compute_overlap(old_bins, new_bins)
    
    # normalization so that overlaps measure the percentage of each old cell that overlaps with the new bins
    overlaps /= np.abs(old_deltas[:, None])

    # handle aggregation methods that use a modified overlaps matrix
    if aggregation == RegridType.interp_or_mean:
        overlaps = get_interp_or_mean_overlaps(overlaps, old_coords, new_coords)
    elif aggregation == RegridType.nearest_or_mode:
        overlaps = get_nearest_or_mode_overlaps(overlaps, old_coords, new_coords)

    # ensure the dimension being operated on is the last one
    original_dim_idx = data.dims.index(dim)
    if original_dim_idx != len(data.dims) - 1:
        old_data = np.moveaxis(old_data, original_dim_idx, -1)

    # crop out of bounds data from old data/overlaps
    unused_rows = np.all(overlaps == 0, axis=1)
    old_data = old_data[..., ~unused_rows]
    overlaps = overlaps[~unused_rows]

    #ensure the data is contiguous in memory
    old_data = np.ascontiguousarray(old_data)
    overlaps = np.ascontiguousarray(overlaps)

    #set any data that is nan to 0, and keep track of where the nans were
    validmask = ~np.isnan(old_data)
    old_data[~validmask] = 0

    # hacky way to deal with nans not propagating correctly under mode aggregation
    # TODO: is this correct for nearest interpolation?
    if aggregation == RegridType.mode or aggregation == RegridType.nearest_or_mode:
        if not np.all(validmask):
            old_data[~validmask] = float('-inf') #TODO: this line will fail on integer data. but validmask should be all True in that case, so probably no problem...

    #perform the regridding on the data, and replace any nans
    result = regrid_1d_reducer(old_data, overlaps, aggregation, low_memory)
    replace_nans(result, validmask, overlaps)

    # move the dimension back to its original position
    result = np.moveaxis(result, -1, original_dim_idx)
    result = np.ascontiguousarray(result)

    #convert time coords back to cftime.DatetimeNoLeap
    if dim == 'time':
        new_coords = new_coords_copy

    #convert back to xarray, with the new coords
    result = xr.DataArray(result, coords={**data.coords, dim:new_coords}, dims=data.dims)

    return result





def regrid_1d_reducer(old_data:np.ndarray, overlaps:np.ndarray, aggregation:RegridType, low_memory:bool=False) -> np.ndarray:
    """
    Perform the actual regridding reduction over the output bins, according to the aggregation method
    """
    
    # low memory mode uses less memory, but is less accurate
    if low_memory:
        old_data = old_data.astype(np.float32)
        overlaps = overlaps.astype(np.float32)

    overlap_mask = overlaps > 0
    cols = np.any(overlap_mask, axis=0)
    starts = overlap_mask.argmax(axis=0)
    ends = overlap_mask.shape[0] - (overlap_mask[::-1]).argmax(axis=0)

    #determine the length of the longest column
    max_col_length = np.max(ends[cols] - starts[cols])

    #set the starts and ends of the empty columns to 0:max_col_length
    starts[~cols] = 0
    ends[~cols] = max_col_length

    #extend all columns so that they are of length max_col_length
    ends[cols] = (starts[cols] + max_col_length).clip(max=overlap_mask.shape[0])
    starts[cols] = (ends[cols] - max_col_length).clip(min=0)

    #convert overlaps/overlap_mask so that unselected locations are nan
    overlaps[~overlap_mask] = np.nan
    overlap_mask[~overlap_mask] = np.nan

    # set up selectors for the indices of each bin to aggregate over
    col_selector = np.arange(max_col_length) + starts[:, None]
    row_selector = np.arange(overlap_mask.shape[1])[:, None]

    #construct a matrix holding all the input values for each bin in the output
    unmasked_binned_data = old_data[..., col_selector]
    
    if aggregation in float_weighted_reduction_methods:
        # weight values in the bin before aggregating
        bin_mask = overlaps[col_selector, row_selector]
    else:
        # take all values in the bin unmodified for aggregation
        bin_mask = overlap_mask[col_selector, row_selector]
    
    # mask (or weight) the sets of values in the bins
    binned_data = unmasked_binned_data * bin_mask

    # perform reduction over bins according to aggregation method
    if aggregation == RegridType.min:
        result = np.nanmin(binned_data, axis=-1)
    elif aggregation == RegridType.max:
        result = np.nanmax(binned_data, axis=-1)
    elif aggregation == RegridType.mean or aggregation == RegridType.interp_or_mean:
        result = np.nansum(binned_data, axis=-1) / np.nansum(bin_mask, axis=-1)
    elif aggregation == RegridType.median:
        result = np.nanmedian(binned_data, axis=-1)
    elif aggregation == RegridType.mode:
        result = stats.mode(binned_data, axis=-1, nan_policy='omit', keepdims=False)[0]
    elif aggregation == RegridType.nearest_or_mode:
        if binned_data.shape[-1] == 1:  # select the only value in each bin
            result = binned_data[..., 0]
        else:
            result = stats.mode(binned_data, axis=-1, nan_policy='omit', keepdims=False)[0]
    elif aggregation == RegridType.conserve:
        result = np.nansum(binned_data, axis=-1)
    else:
        raise NotImplementedError(f'Unrecognized regrid aggregation method: {aggregation}. Expected one of: {[*RegridType.__members__.values()]}')

    return result
    

def replace_nans(result:np.ndarray, validmask:np.ndarray, overlaps:np.ndarray):
    if np.all(validmask):
        return

    #TODO: handling MemoryError when running out of memory
    nan_accumulation = validmask.astype(np.float32) @ (overlaps > 0).astype(np.float32)
    result[nan_accumulation == 0] = np.nan
