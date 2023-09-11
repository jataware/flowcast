from __future__ import annotations

import numpy as np
import torch
from torch.cuda import OutOfMemoryError
import xarray as xr
from dynamic_insights import Pipeline, Realization, Scenario, CMIP6Data, OtherData, Threshold, ThresholdType, Model
from matplotlib import pyplot as plt
from enum import Enum, auto
from typing import Callable
import pdb





def main():
    pipe = Pipeline(
        realizations=Realization.r1i1p1f1, 
        scenarios=[Scenario.ssp126, Scenario.ssp245, Scenario.ssp370, Scenario.ssp585]
    )

    pipe.load('pop', OtherData.population)
    pipe.load('land_cover', OtherData.land_cover)
    pipe.load('tasmax', CMIP6Data.tasmax, Model.CAS_ESM2_0)
    pipe.execute()
    pop = pipe.get_value('pop').data
    modis = pipe.get_value('land_cover').data
    tasmax = pipe.get_value('tasmax').data


    #conservatively convert population to the resolution of modis
    new_pop = regrid_1d(pop, modis['lat'].data, 'lat', aggregation=Aggregation.conserve)
    new_pop = regrid_1d(new_pop, modis['lon'].data, 'lon', aggregation=Aggregation.conserve)

    #plot old vs new population (clip old population to match bounds of new population)
    plt.figure()
    old_pop = pop.isel(time=0,ssp=-1).sel(lat=slice(modis['lat'].max(), modis['lat'].min()), lon=slice(modis['lon'].min(), modis['lon'].max()))
    plt.imshow(np.log(old_pop + 1e-6))
    # plt.imshow(old_pop)

    plt.figure()
    plt.imshow(np.log(new_pop.isel(time=0,ssp=-1) + 1e-6))
    # plt.imshow(new_pop.isel(time=0,ssp=-1))

    plt.show()
    
    pdb.set_trace()


    #conservatively convert population to the resolution of modis
    new_pop = regrid_1d(pop, tasmax['lat'].data, 'lat', aggregation=Aggregation.conserve)
    new_pop = regrid_1d(new_pop, tasmax['lon'].data, 'lon', aggregation=Aggregation.conserve)

    #plot old vs new population (clip old population to match bounds of new population)
    plt.figure()
    old_pop = pop.isel(time=0,ssp=-1).sel(lat=slice(tasmax['lat'].min(), tasmax['lat'].max(),-1), lon=slice(modis['lon'].min(), modis['lon'].max()))
    # plt.imshow(np.log(old_pop + 1e-6))
    plt.imshow(old_pop)

    plt.figure()
    # plt.imshow(np.log(new_pop.isel(time=0,ssp=-1) + 1e-6))
    plt.imshow(new_pop.isel(time=0,ssp=-1))

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
    #nearest = auto() #TODO: figure out how this would work. probably use a different regridding method entirely




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
    
    return np.ascontiguousarray(overlap)



def get_bins(coords:np.ndarray, offset:BinOffset) -> np.ndarray:
    """
    Given an array of evenly spaced coordinates, return the bin edges

    Parameters:
        coords (np.ndarray): The array of evenly spaced coordinates
        offset (BinOffset): Whether the bins should be left, centered, or right of the coordinates

    Returns:
        np.ndarray: The bin edges
    """
    delta = coords[1] - coords[0]
    assert np.allclose(np.diff(coords), delta), f'coords must be evenly spaced. Got spacings: {np.diff(coords)}'
    if offset == BinOffset.left:
        bins = np.linspace(coords[0] - delta, coords[-1], len(coords) + 1)
    elif offset == BinOffset.center:
        bins = np.linspace(coords[0] - delta/2, coords[-1] + delta/2, len(coords) + 1)
    elif offset == BinOffset.right:
        bins = np.linspace(coords[0], coords[-1] + delta, len(coords) + 1)
    return bins




def regrid_1d(
        data:xr.DataArray,
        new_coords:np.ndarray,
        dim:str,
        offset:BinOffset=BinOffset.left,
        aggregation=Aggregation.mean,
        wrap:tuple[float,float]=None,
        try_gpu:bool=True
    ) -> xr.DataArray:
    
    old_coords = data[dim].data
    old_data = data.data.copy()

    # get the bin boundaries for the old and new data
    old_bins = get_bins(old_coords, offset)
    new_bins = get_bins(new_coords, offset)
    

    #compute the amount of overlap between each old bin and each new bin
    overlaps = compute_overlap(old_bins, new_bins)
    
    # normalization....TODO: adjust based on aggregation method
    old_delta = old_bins[1] - old_bins[0]
    overlaps /= np.abs(old_delta)

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

    #do the matrix multiplication on the gpu if possible
    if try_gpu and torch.cuda.is_available():
        result = regrid_1d_general_accumulate_gpu(old_data, overlaps, aggregation)
        replace_nans_gpu(result, validmask, overlaps)
    else:
        result = regrid_1d_general_accumulate_cpu(old_data, overlaps, aggregation)
        replace_nans_cpu(result, validmask, overlaps)

    # move the dimension back to its original position
    result = np.moveaxis(result, -1, original_dim_idx)
    result = np.ascontiguousarray(result)


    #convert back to xarray, with the new coords
    result = xr.DataArray(result, coords={**data.coords, dim:new_coords}, dims=data.dims)

    return result


def regrid_1d_conserve_accumulate_gpu(old_data:np.ndarray, overlaps:np.ndarray) -> np.ndarray:
    try:
        # move data to the gpu
        gpu_old_data = torch.tensor(old_data, device='cuda')
        gpu_overlaps = torch.tensor(overlaps, device='cuda')
        
        # do the regridding, and move the result off the gpu to free up memory
        gpu_result = gpu_old_data @ gpu_overlaps
        result = gpu_result.cpu().numpy()
        del gpu_result, gpu_old_data, gpu_overlaps

        return result

        
    except RuntimeError as e:
        if 'CUDA out of memory' not in str(e):
            raise e
        pdb.set_trace()
        ...
    
    
    pdb.set_trace()
    ...

def regrid_1d_conserve_accumulate_cpu(old_data:np.ndarray, overlaps:np.ndarray) -> np.ndarray:
    try:
        return old_data @ overlaps
    except Exception as e:
        pdb.set_trace()
        ...
    pdb.set_trace()
    ...

def regrid_1d_general_accumulate_gpu(old_data:np.ndarray, overlaps:np.ndarray, aggregation:Aggregation) -> np.ndarray:
    if aggregation == Aggregation.conserve:
        return regrid_1d_conserve_accumulate_gpu(old_data, overlaps)
    pdb.set_trace()
    ...

def regrid_1d_general_accumulate_cpu(old_data:np.ndarray, overlaps:np.ndarray, aggregation:Aggregation) -> np.ndarray:
    if aggregation == Aggregation.conserve:
        return regrid_1d_conserve_accumulate_cpu(old_data, overlaps)
    pdb.set_trace()
    ...

def replace_nans_gpu(result:np.ndarray, validmask:np.ndarray, overlaps:np.ndarray):
    if np.all(validmask):
        return
    try:
        # compute the new locations of nans, and move the result off the gpu to free up memory
        gpu_overlaps = torch.tensor(overlaps, device='cuda')
        gpu_validmask = torch.tensor(validmask, device='cuda', dtype=torch.float32)

        gpu_nan_accumulation = gpu_validmask @ (gpu_overlaps > 0).to(gpu_validmask.dtype)
        nan_accumulation = gpu_nan_accumulation.cpu().numpy()
        del gpu_nan_accumulation, gpu_overlaps, gpu_validmask

        # replace any masked nans with the original nans
        result[nan_accumulation == 0] = np.nan
        return
    
    except RuntimeError as e:
        if 'CUDA out of memory' not in str(e):
            raise e
        pdb.set_trace()
        ...

    pdb.set_trace()
    ...

def replace_nans_cpu(result:np.ndarray, validmask:np.ndarray, overlaps:np.ndarray):
    if np.all(validmask):
        return
    try:
        nan_accumulation = validmask @ overlaps #TODO: is this correct?
        result[nan_accumulation == 0] = np.nan
        return
    except Exception as e:
        pdb.set_trace()
        ...
    pdb.set_trace()
    ...


# # map from (torch.cuda.is_available(), Aggregation) to aggregator function
# regrid_gpu_matmul_dispatch: dict[Aggregation, Callable] = {
#     Aggregation.conserve: regrid_1d_conserve_gpu,


# }

# regrid_cpu_matmul_dispatch: dict[Aggregation, Callable] = {
#     Aggregation.conserve: regrid_1d_conserve_cpu,


# }




if __name__ == '__main__':
    main()