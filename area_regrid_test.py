from __future__ import annotations

import numpy as np
import xarray as xr
from dynamic_insights import Pipeline, Realization, Scenario, CMIP6Data, OtherData, Model, determine_longitude_convention, convert_longitude_convention, LongitudeConvention
from matplotlib import pyplot as plt
from enum import Enum, auto
from warnings import warn
import pdb


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


from cftime import DatetimeNoLeap
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


def main():
    pipe = Pipeline(
        realizations=Realization.r1i1p1f1, 
        scenarios=[Scenario.ssp585]#[Scenario.ssp126, Scenario.ssp245, Scenario.ssp370, Scenario.ssp585]
    )

    pipe.load('pop', OtherData.population)
    pipe.load('land_cover', OtherData.land_cover)
    pipe.load('tasmax', CMIP6Data.tasmax, Model.CAS_ESM2_0)
    pipe.load('pr', CMIP6Data.pr, Model.FGOALS_f3_L)
    pipe.load('tas', CMIP6Data.tas, Model.FGOALS_f3_L)
    pipe.execute()
    pop = pipe.get_value('pop').data
    modis = pipe.get_value('land_cover').data
    tasmax = pipe.get_value('tasmax').data
    pr = pipe.get_value('pr').data
    tas = pipe.get_value('tas').data


    #ensure the lon conventions match
    inplace_set_longitude_convention(pop, LongitudeConvention.neg180_180)
    inplace_set_longitude_convention(modis, LongitudeConvention.neg180_180)
    inplace_set_longitude_convention(tasmax, LongitudeConvention.neg180_180)
    inplace_set_longitude_convention(pr, LongitudeConvention.neg180_180)
    inplace_set_longitude_convention(tas, LongitudeConvention.neg180_180)

    

    # convert pr to yearly, and then the geo resolution of population
    time_axis = np.array([DatetimeNoLeap(year, 1, 1) for year in range(2015, 2101)])
    new_pr = regrid_1d(pr, time_axis, 'time', aggregation=Aggregation.interp_mean)
    new_pr = regrid_1d(new_pr, pop['lat'].data, 'lat', aggregation=Aggregation.interp_mean)
    new_pr = regrid_1d(new_pr, pop['lon'].data, 'lon', aggregation=Aggregation.interp_mean)
    plt.figure()
    plt.imshow(new_pr.isel(time=0,ssp=-1, realization=0))
    plt.figure()
    plt.imshow(pr.isel(time=0,ssp=-1, realization=0), origin='lower')
    plt.show()



    #convert tas to yearly, and then the geo resolution of population
    time_axis = np.array([DatetimeNoLeap(year, 1, 1) for year in range(2015, 2101)])
    new_tas = regrid_1d(tas, time_axis, 'time', aggregation=Aggregation.interp_mean)
    new_tas = regrid_1d(new_tas, pop['lat'].data, 'lat', aggregation=Aggregation.interp_mean)
    new_tas = regrid_1d(new_tas, pop['lon'].data, 'lon', aggregation=Aggregation.interp_mean)
    plt.figure()
    plt.imshow(new_tas.isel(time=0,ssp=-1, realization=0))
    plt.figure()
    plt.imshow(tas.isel(time=0,ssp=-1, realization=0), origin='lower')
    plt.show()



    #convert tasmax to yearly, and then the geo resolution of population
    time_axis = np.array([DatetimeNoLeap(year, 1, 1) for year in range(2015, 2101)])
    new_tasmax = regrid_1d(tasmax, time_axis, 'time', aggregation=Aggregation.interp_mean)
    new_tasmax = regrid_1d(new_tasmax, pop['lat'].data, 'lat', aggregation=Aggregation.max)
    new_tasmax = regrid_1d(new_tasmax, pop['lon'].data, 'lon', aggregation=Aggregation.max)
    plt.figure()
    plt.imshow(new_tasmax.isel(time=0,ssp=-1, realization=0))
    plt.figure()
    plt.imshow(tasmax.isel(time=0,ssp=-1, realization=0), origin='lower')
    plt.show()

    
    #conservatively convert population to the resolution of modis
    new_pop = regrid_1d(pop, modis['lat'].data, 'lat', aggregation=Aggregation.conserve)
    new_pop = regrid_1d(new_pop, modis['lon'].data, 'lon', aggregation=Aggregation.conserve)

    #plot old vs new population (clip old population to match bounds of new population)
    plt.figure()
    old_pop = pop.isel(time=0,ssp=-1).sel(lat=slice(new_pop['lat'].max(), new_pop['lat'].min()), lon=slice(new_pop['lon'].min(), new_pop['lon'].max()))
    plt.imshow(np.log(old_pop + 1e-6))
    # plt.imshow(old_pop)

    plt.figure()
    plt.imshow(np.log(new_pop.isel(time=0,ssp=-1) + 1e-6))
    # plt.imshow(new_pop.isel(time=0,ssp=-1))
    
    old_pop_count = np.nansum(old_pop)
    new_pop_count = np.nansum(new_pop.isel(time=0,ssp=-1))
    print(f'comparing pop count: {old_pop_count=} vs {new_pop_count=}. Error: {np.abs(old_pop_count - new_pop_count) / old_pop_count:.4%}')

    plt.show()
    

    #conservatively convert population to the resolution of modis
    new_pop = regrid_1d(pop, tasmax['lat'].data, 'lat', aggregation=Aggregation.conserve)
    new_pop = regrid_1d(new_pop, tasmax['lon'].data, 'lon', aggregation=Aggregation.conserve)

    #plot old vs new population (clip old population to match bounds of new population)
    plt.figure()
    old_pop = pop.isel(time=0,ssp=-1).sel(lat=slice(new_pop['lat'].min(), new_pop['lat'].max(),-1), lon=slice(new_pop['lon'].min(), new_pop['lon'].max()))
    plt.imshow(np.log(old_pop + 1e-6), origin='lower')
    # plt.imshow(old_pop, origin='lower')

    plt.figure()
    plt.imshow(np.log(new_pop.isel(time=0,ssp=-1) + 1e-6), origin='lower')
    # plt.imshow(new_pop.isel(time=0,ssp=-1), origin='lower')

    old_pop_count = np.nansum(old_pop)
    new_pop_count = np.nansum(new_pop.isel(time=0,ssp=-1))
    print(f'comparing pop count: {old_pop_count=} vs {new_pop_count=}. Error: {np.abs(old_pop_count - new_pop_count) / old_pop_count:.4%}')

    plt.show()




    pdb.set_trace()
    ...


class BinOffset(Enum):
    left = auto()
    center = auto()
    right = auto()

class Aggregation(Enum):
    conserve = auto()
    min = auto()
    max = auto()
    mean = auto()
    median = auto()      #TODO: take the median of each bin
    mode = auto()        #TODO: take the mode over the bin
    interp_mean = auto() #TODO: interp if increasing resolution, mean if decreasing resolution
    nearest = auto()     #TODO: take the centermost item of each bin




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

def get_interp_mean_overlaps(overlaps:np.ndarray, offset:BinOffset, old_coords:np.ndarray, new_coords:np.ndarray) -> np.ndarray:
    """
    Convert the overlaps matrix to one that will perform:
    - interpolation for resolution increases
    - mean aggregation for resolution decreases
    """

    # determine if the resolution is increasing or decreasing
    nonzero_indices = np.nonzero(overlaps)
    old_size = nonzero_indices[0].max() - nonzero_indices[0].min()
    new_size = nonzero_indices[1].max() - nonzero_indices[1].min()
    if new_size <= old_size:
        # resolution decrease just uses existing overlaps matrix
        return overlaps

    # create a mask over which values are included in the interpolation for that bin
    interp_mask = overlaps.copy()
    if offset == BinOffset.left:
        interp_mask[:-1] += overlaps[1:]
    elif offset == BinOffset.center:
        interp_mask[:-1] += overlaps[1:]
        interp_mask[1:] += overlaps[:-1]
    elif offset == BinOffset.right:
        interp_mask[1:] += overlaps[:-1]
    interp_mask = interp_mask > 0

    # compute the distances of each old cell from the new cell (masking those that are too far away)
    offsets = np.abs(old_coords[:,None] - new_coords[None, :]) * interp_mask

    # invert the offsets so that the closest cell has the largest weight
    offsets = (offsets.sum(axis=0)[None] - offsets) * interp_mask

    # normalize so that each column has the same sum as the original column
    offset_sum = offsets.sum(axis=0)
    offsets[:, offset_sum > 0] /= offset_sum[offset_sum > 0]
    offsets *= overlaps.sum(axis=0)[None]

    return offsets



def get_bins(coords:np.ndarray, offset:BinOffset) -> np.ndarray:
    """
    Given an array of evenly spaced coordinates, return the bin edges

    Parameters:
        coords (np.ndarray): The array of evenly spaced coordinates
        offset (BinOffset): Whether the bins should be left, centered, or right of the coordinates

    Returns:
        np.ndarray, np.ndarray: The bin edges, and the deltas between each bin
    """
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
        aggregation=Aggregation.interp_mean,
        wrap:tuple[float,float]=None, #TBD format for this...
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

    if aggregation == Aggregation.interp_mean:
        # modify the overlaps matrix so that it works for interp_mean
        overlaps = get_interp_mean_overlaps(overlaps, offset, old_coords, new_coords)
        aggregation = Aggregation.mean
    #TODO: may need to have a similar process for mode...

    # ensure the dimension being operated on is the last one
    original_dim_idx = data.dims.index(dim)
    if original_dim_idx != len(data.dims) - 1:
        old_data = np.moveaxis(old_data, original_dim_idx, -1)

    #ensure the data is contiguous in memory
    old_data = np.ascontiguousarray(old_data)
    overlaps = np.ascontiguousarray(overlaps)

    #set any data that is nan to 0, and keep track of where the nans were
    validmask = ~np.isnan(old_data)
    old_data[~validmask] = 0

    #perform the regridding on the data, and replace any nans
    result = regrid_1d_reducer(old_data, overlaps, aggregation)
    replace_nans(result, validmask, overlaps)

    # move the dimension back to its original position
    result = np.moveaxis(result, -1, original_dim_idx)
    result = np.ascontiguousarray(result)

    #convert time coords back to cftime.DatetimeNoLeap
    if dim == 'time':
        new_coords = new_coords_copy

    #convert back to xarray, with the new coords
    result = xr.DataArray(result, coords={**data.coords, dim:new_coords}, dims=data.dims)

    print(f'result shape: {result.shape}')
    return result



def regrid_1d_reducer(old_data:np.ndarray, overlaps:np.ndarray, aggregation:Aggregation) -> np.ndarray:
    """
    Perform the actual regridding reduction over the output bins, according to the aggregation method
    """
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
    
    # perform reduction over bins according to aggregation method
    if aggregation == Aggregation.min:
        bin_mask = overlap_mask[col_selector, row_selector]
        binned_data = unmasked_binned_data * bin_mask
        result = np.nanmin(binned_data, axis=-1)

    elif aggregation == Aggregation.max:
        bin_mask = overlap_mask[col_selector, row_selector]
        binned_data = unmasked_binned_data * bin_mask
        result = np.nanmax(binned_data, axis=-1)

    elif aggregation == Aggregation.mean:
        bin_mask = overlaps[col_selector, row_selector]
        binned_data = unmasked_binned_data * bin_mask
        result = np.nansum(binned_data, axis=-1) / np.nansum(bin_mask, axis=-1)

    elif aggregation == Aggregation.median:
        bin_mask = overlap_mask[col_selector, row_selector]
        binned_data = unmasked_binned_data * bin_mask
        result = np.nanmedian(binned_data, axis=-1)

    elif aggregation == Aggregation.mode:
        pdb.set_trace()
        raise NotImplementedError(f'Aggregation method {aggregation} not implemented.')
        # bin_mask = overlap_mask[col_selector, row_selector]
        # binned_data = unmasked_binned_data * bin_mask
        # result = # np.nanmode isn't a real function...

    elif aggregation == Aggregation.nearest:
        pdb.set_trace()
        raise NotImplementedError(f'Aggregation method {aggregation} not implemented.')
        #TODO: modify overlaps/overlap_mask to just select the centermost item in each bin
        # bin_mask = overlap_mask[col_selector, row_selector]
        # binned_data = unmasked_binned_data * bin_mask

    elif aggregation == Aggregation.conserve:
        bin_mask = overlaps[col_selector, row_selector]
        binned_data = unmasked_binned_data * bin_mask
        result = np.nansum(binned_data, axis=-1)

    else:
        raise NotImplementedError(f'Aggregation method {aggregation} not implemented.')

    return result
    

def replace_nans(result:np.ndarray, validmask:np.ndarray, overlaps:np.ndarray):
    if np.all(validmask):
        return

    #TODO: handling MemoryError when running out of memory
    nan_accumulation = validmask.astype(np.float32) @ (overlaps > 0).astype(np.float32)
    result[nan_accumulation == 0] = np.nan






if __name__ == '__main__':
    main()