from __future__ import annotations

import xarray as xr
import numpy as np
# from flowcast.pipeline import Pipeline, Variable, Threshold, ThresholdType
from debugging_pipeline_copy import Pipeline, Variable, Threshold, ThresholdType
from flowcast.regrid import RegridType as TimeRegridType, RegridType as GeoRegridType
from flowcast.spacetime import Frequency
from pathlib import Path
from matplotlib import pyplot as plt, animation
import imageio


import pdb

"""
pip install imageio[ffmpeg]
"""

"""
CMIP precip long range thresholded to “drought” levels
MODIS thresholded to rural land area
asset wealth thresholded to bottom quartile (does this make sense based on that data?)
GPW population
output1 ----> gridded time series of low income, rural population in long term drought prone areas
output2 ---> country level time series of low income, rural population in long term drought prone areas
"""

def get_pr_zscores() -> xr.DataArray:
    # load datasets
    pr:xr.DataArray = xr.open_dataset('pr.nc')['pr']
    pr_baseline = pr.isel(time=slice(12*8-1)) #first 7 years
    pr_mean, pr_std = pr_baseline.mean(dim=('time')), pr_baseline.std(dim=('time'))
    pr_z: xr.DataArray = (pr - pr_mean) / pr_std #standardize
    # pdb.set_trace()

    #save pr_z to netcdf
    pr_z.to_dataset(name='pr_z').to_netcdf('pr_z.nc')

    return pr_z



def generate_data():
    # load datasets
    pop = xr.open_dataset('pop.nc')['population']
    pr_z = get_pr_zscores()
    wealth = xr.open_dataset('wealth.nc')['asset_wealth'].rename({'latitude': 'lat', 'longitude': 'lon'})
    modis = xr.open_dataset('modis.nc')['land_cover']

    pipe = Pipeline(low_memory=True)
    pipe.set_geo_resolution('pop')
    pipe.set_time_resolution(Frequency.yearly)
    # pipe.set_time_resolution('pr_z')
    pipe.load('pop', lambda: Variable(pop, TimeRegridType.interp_or_mean, GeoRegridType.conserve))
    pipe.load('pr_z', lambda: Variable(pr_z, TimeRegridType.min, GeoRegridType.interp_or_mean))
    pipe.load('wealth', lambda: Variable(wealth, TimeRegridType.median, GeoRegridType.median))
    pipe.load('modis', lambda: Variable(modis, TimeRegridType.nearest_or_mode, GeoRegridType.nearest_or_mode))

    #raw precipitation values: 1e-5 is ~35th percentile. 25th percentile is 6.273722647165414e-06
    pipe.threshold('pr_drought_mask', 'pr_z', Threshold(-1, ThresholdType.less_than_or_equal)) 
    pipe.threshold('poverty_mask', 'wealth', Threshold(3, ThresholdType.less_than_or_equal))
    pipe.threshold('rural_mask', 'modis', Threshold(13, ThresholdType.not_equal))

    pipe.multiply('rural_poverty_mask', 'poverty_mask', 'rural_mask')
    pipe.multiply('drought_rural_poverty_mask', 'rural_poverty_mask', 'pr_drought_mask')
    pipe.multiply('affected_pop0', 'pop', 'drought_rural_poverty_mask')

    pipe.reverse_geocode('affected_pop1', 'affected_pop0', ['Ethiopia'])
    pipe.sum_reduce('affected_pop2', 'affected_pop1', ['lat', 'lon'])
    pipe.sum_reduce('affected_latlon', 'affected_pop1', ['admin0'])

    #debugging population
    pipe.reverse_geocode('pop1', 'pop', ['Ethiopia'])

    #save to netcdf
    pipe.save('affected_pop2', 'affected_ethiopia.nc')
    pipe.save('affected_latlon', 'affected_gridded.nc')

    pipe.execute()


    # output data
    x1 = pipe.get_value('affected_pop2').data
    x2 = pipe.get_value('affected_latlon').data

    # plot
    plt.figure()
    x1.plot()
    
    plt.figure()
    x2.isel(time=0).plot()
    
    plt.show()


    #debug dump data to npy
    np.save('affected_latlon.npy', x2.data)

    # #make a video out of the gridded data
    # get_year_string = lambda i: f'year = {str(x1.time.isel(time=i).data.item())[:4]}'
    # make_video(x2.data, Path('gridded.mp4'), 'Poverty Drought Exposure ', get_year_string)

    # pdb.set_trace()

def save_to_video():
    if not Path('affected_latlon.npy').exists():
        generate_data()
    data = np.load('affected_latlon.npy')
    make_video(data, Path('gridded.mkv'), 'Poverty Drought Exposure ')

def make_video(data:np.ndarray, save_path:Path, title:str='', get_year_string=lambda x: f'year {(x-1)//12+2015}'):
    #make a video out of each frame of the data
    #then save the video to the save_path

    # convert the data to a 4D array with the last dimension being the color channels
    data = data / np.max(data)
    cmap = plt.get_cmap('viridis')
    data = cmap(data)

    #convert to uint8
    data = (data * 255).astype(np.uint8)

    #save to video
    imageio.mimsave(save_path, data, fps=30, codec='libx265', pixelformat='yuv444p', output_params=['-crf', '0', '-preset', 'veryslow', '-x265-params', 'lossless=1'])

    
    #save to video with matplotlib
    # fix, ax = plt.subplots()

    # def update(i:int):
    #     ax.clear()
    #     ax.imshow(data[i])
    #     ax.set_title(title + get_year_string(i))
    #     ax.set_xticks([])
    #     ax.set_yticks([])

    # anim = animation.FuncAnimation(fix, update, frames=data.shape[0], interval=100)
    # anim.save(save_path, writer='ffmpeg')



if __name__ == '__main__':
    generate_data()
    # save_to_video()