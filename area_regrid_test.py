from __future__ import annotations

import numpy as np
import torch
import xarray as xr
from dynamic_insights import Pipeline, Realization, Scenario, CMIP6Data, OtherData, Threshold, ThresholdType, Model
from matplotlib import pyplot as plt
from enum import Enum, auto
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
    new_pop = regrid_1d(pop, modis['lat'].data, 'lat', aggregation=Aggregation.add)
    new_pop = regrid_1d(new_pop, modis['lon'].data, 'lon', aggregation=Aggregation.add)

    #plot old vs new population (clip old population to match bounds of new population)
    plt.figure()
    old_pop = pop.isel(time=0,ssp=-1).sel(lat=slice(modis['lat'].min(), modis['lat'].max(),-1), lon=slice(modis['lon'].min(), modis['lon'].max()))
    plt.imshow(np.log(old_pop + 1e-6), origin='lower')

    plt.figure()
    plt.imshow(np.log(new_pop.isel(time=0,ssp=-1) + 1e-6), origin='lower')

    plt.show()
    
    pdb.set_trace()


    #conservatively convert population to the resolution of modis
    new_pop = regrid_1d(pop, tasmax['lat'].data, 'lat', aggregation=Aggregation.add)
    new_pop = regrid_1d(new_pop, tasmax['lon'].data, 'lon', aggregation=Aggregation.add)

    #plot old vs new population (clip old population to match bounds of new population)
    plt.figure()
    old_pop = pop.isel(time=0,ssp=-1).sel(lat=slice(tasmax['lat'].min(), tasmax['lat'].max(),-1), lon=slice(modis['lon'].min(), modis['lon'].max()))
    plt.imshow(np.log(old_pop + 1e-6), origin='lower')

    plt.figure()
    plt.imshow(np.log(new_pop.isel(time=0,ssp=-1) + 1e-6), origin='lower')

    plt.show()




    pdb.set_trace()
    ...


class BinOffset(Enum):
    left = auto()
    center = auto()
    right = auto()

class Aggregation(Enum):
    add = auto()
    min = auto()
    max = auto()
    mean = auto()
    #nearest = auto() #TODO: figure out how this would work. probably use a different regridding method entirely




# def compute_overlap(a:np.ndarray, b:np.ndarray) -> np.ndarray:
#     a = a[:, None]

#     if a[1] > a[0]:
#         overlap = np.maximum(0, np.minimum(a[1:], b[1:]) - np.maximum(a[:-1], b[:-1]))
#     else:
#         overlap = np.maximum(0, np.minimum(a[:-1], b[:-1]) - np.maximum(a[1:], b[1:]))
    
#     return overlap


def compute_overlap(a: np.ndarray, b: np.ndarray) -> np.ndarray:
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



def regrid_1d(
        data:xr.DataArray, 
        new_coords:np.ndarray, 
        dim:str, 
        offset:BinOffset=BinOffset.left, 
        aggregation=Aggregation.mean, 
        wrap:tuple[float,float]=None
    ) -> xr.DataArray:
    
    old_coords = data[dim].data
    old_data = data.data.copy()


    
    old_delta = old_coords[1] - old_coords[0]
    new_delta = new_coords[1] - new_coords[0]
    assert np.allclose(np.diff(old_coords), old_delta), f'old coords must be evenly spaced. Got spacings: {np.diff(old_coords)}'
    assert np.allclose(np.diff(new_coords), new_delta), f'new coords must be evenly spaced. Got spacings: {np.diff(new_coords)}'

    if offset == BinOffset.left:
        old_bins = np.linspace(old_coords[0] - old_delta, old_coords[-1], len(old_coords) + 1)
        new_bins = np.linspace(new_coords[0] - new_delta, new_coords[-1], len(new_coords) + 1)
    elif offset == BinOffset.center:
        old_bins = np.linspace(old_coords[0] - old_delta/2, old_coords[-1] + old_delta/2, len(old_coords) + 1)
        new_bins = np.linspace(new_coords[0] - new_delta/2, new_coords[-1] + new_delta/2, len(new_coords) + 1)
    elif offset == BinOffset.right:
        old_bins = np.linspace(old_coords[0], old_coords[-1] + old_delta, len(old_coords) + 1)
        new_bins = np.linspace(new_coords[0], new_coords[-1] + new_delta, len(new_coords) + 1)


    #compute the amount of overlap between each old bin and each new bin
    overlaps = compute_overlap(old_bins, new_bins)
    overlaps /= np.abs(old_delta)

    # old_idx, new_idx = np.where(overlaps)
    # proportions = overlaps[old_idx, new_idx]


    # ensure the dimension being operated on is the last one
    original_dim_idx = data.dims.index(dim)
    if original_dim_idx != len(data.dims) - 1:
        old_data = np.moveaxis(old_data, original_dim_idx, -1)

    #set any data that is nan to 0, and keep track of where the nans were
    validmask = ~np.isnan(old_data)
    old_data[~validmask] = 0

    #do the matrix multiplication on the gpu if possible
    if torch.cuda.is_available():
        old_data = torch.tensor(old_data, device='cuda')
        overlaps = torch.tensor(overlaps, device='cuda')
    else:
        old_data = np.ascontiguousarray(old_data)
        overlaps = np.ascontiguousarray(overlaps)

    # do the regridding #TODO: handling accumulators other than +
    result = old_data @ overlaps
    
    if torch.cuda.is_available():
        gpu_res = result
        result = result.cpu().numpy()
        del gpu_res

    # replace any masked nans
    if torch.cuda.is_available():
        validmask = torch.tensor(validmask, device='cuda', dtype=torch.float16)
        overlaps  = torch.tensor(overlaps > 0, device='cuda', dtype=torch.float16)
    else:
        validmask = np.ascontiguousarray(validmask)
    
    nan_accumulation = validmask @ overlaps
    
    if torch.cuda.is_available():
        gpu_nan = nan_accumulation
        nan_accumulation = nan_accumulation.cpu().numpy()
        del gpu_nan

    result[nan_accumulation == 0] = np.nan
    

    # move the dimension back to its original position
    result = np.moveaxis(result, -1, original_dim_idx)
    result = np.ascontiguousarray(result)


    #convert back to xarray, with the new coords
    # new_coords = xr.DataArray(new_coords, dims=[dim])
    result = xr.DataArray(result, coords={**data.coords, dim:new_coords}, dims=data.dims)

    return result

    # # return xr.DataArray(result, coords=data.coords, dims=data.dims)

    # # res = overlaps[..., None] # old_data[...,]

    # pdb.set_trace()


    # shape = data.shape[:-1] + (len(new_coords),)
    # new_data = np.zeros(shape, dtype=data.dtype)
    # accumulations = np.zeros(shape, dtype=np.float64)

    # idx = [np.arange(s) for s in data.shape[:-1]] + [new_idx]



    # # idx = [np.arange(s) for s in data.shape[:-2]] + [lat_idx, lon_idx]
    # # mesh = np.meshgrid(*idx, indexing='ij')
    # # np.add.at(new_data, tuple(mesh), data)

    # pdb.set_trace()
    # ...


if __name__ == '__main__':
    main()