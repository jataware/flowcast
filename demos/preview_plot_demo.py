from __future__ import annotations

import xarray as xr
import numpy as np
from flowcast.pipeline import Pipeline, Variable, Threshold, ThresholdType
# from debugging_pipeline_copy import Pipeline, Variable, Threshold, ThresholdType
from flowcast.regrid import RegridType as TimeRegridType, RegridType as GeoRegridType
from flowcast.spacetime import Frequency, points_to_mask
from data import OtherData, Scenario
from tmp_data_modeling_previews import generate_preview, plot_png64 #these are from dojo
import pandas as pd
from matplotlib import pyplot as plt

import pdb

pop_loader = OtherData.population(scenario=Scenario.ssp585)

# load earthquake data as lat/lon points, and convert to a gridded mask
quake_df = pd.read_csv('https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_month.csv')
quake_lats, quake_lons = quake_df['latitude'].values, quake_df['longitude'].values
quake_spots = points_to_mask(quake_lats, quake_lons, n_lat=1800, n_lon=3600)
quake_loader = lambda: Variable(quake_spots, geo_regrid_type=GeoRegridType.nearest_or_mode, time_regrid_type=None)


pop_without_time = pop_loader().data.isel(time=0).drop('time')

def main():
    pipe = Pipeline()
    pipe.load('pop', pop_loader)
    # pipe.load('pop', lambda: Variable(pop_without_time, geo_regrid_type=GeoRegridType.nearest_or_mode, time_regrid_type=TimeRegridType.interp_or_mean))
    
    # pipe.load('quake', quake_loader)
    # pipe.mask_to_distance_field('q1', 'quake', include_initial_points=False)
    # pipe.scalar_multiply('q2', 'q1', scalar=-1)
    # pipe.reverse_geocode('q3', 'q2', ['Canada'])#['United States', 'Canada', 'México', 'China', 'Russia', 'Brazil', 'Australia', 'India', 'Argentina', 'Kazakhstan'])

    # pipe.sum_reduce('none', 'pop', ['lat', 'lon'])
    # pipe.sum_reduce('lat', 'pop', ['lon'])
    # pipe.sum_reduce('lon', 'pop', ['lat'])
    pipe.scalar_add('lat,lon', 'pop', scalar=0)
    pipe.reverse_geocode('lat,lon,admin0', 'pop', ['Japan'])#['United States', 'Canada', 'México', 'China', 'Russia', 'Brazil', 'Australia', 'India', 'Argentina', 'Kazakhstan'])
    # pipe.sum_reduce('admin0', 'lat,lon,admin0', ['lat', 'lon'])
    # pipe.sum_reduce('lat,admin0', 'lat,lon,admin0', ['lon'])
    # pipe.sum_reduce('lon,admin0', 'lat,lon,admin0', ['lat'])

    # focus on just africa
    # pipe.reverse_geocode('africa,lat,lon,admin0', 'pop', ['Ethiopia'])#, 'Nigeria', 'Egypt', 'Democratic Republic of the Congo', 'Tanzania', 'South Africa', 'Kenya', 'Uganda', 'Sudan', 'Algeria'])
    # pipe.sum_reduce('africa,admin0', 'africa,lat,lon,admin0', ['lat', 'lon'])
    # pipe.sum_reduce('africa,lat,lon', 'africa,lat,lon,admin0', ['admin0'])

    #TODO: each type of dimension set + plot each preview

    pipe.execute()

    no_time_ids = [
    # 'lat',
    # 'lon',
    'lat,lon',
    # 'admin0',
    'lat,lon,admin0',
    # 'lat,admin0',
    # 'lon,admin0',
    ]
    for id in no_time_ids:
        prev = generate_preview(pipe.env[id].data)#.isel(time=slice(0,5)))
        # plot_png64(prev['log_preview'][0])
        display_video(prev['log_preview'])
    

    # time_ids = ['none']
    # for id in time_ids:
    #     prev = generate_preview(pipe.env[id].data)
    #     plot_png64(prev['log_preview'][0])
    #     plot_png64(prev['preview'][0])
    


    # quake_ids = [
    #     'quake',
    #     'q1',
    #     'q2',
    #     'q3',
    # ]
    # for id in quake_ids:
    #     prev = generate_preview(pipe.env[id].data)
    #     plot_png64(prev['log_preview'][0])
    


    # africa_ids = [
    #     'africa,lat,lon',
    #     'africa,admin0',
    #     'africa,lat,lon,admin0'
    # ]
    # for id in africa_ids:
    #     prev = generate_preview(pipe.env[id].data.isel(time=0))
    #     plot_png64(prev['log_preview'][0])




import base64
from io import BytesIO
from PIL import Image
from matplotlib.animation import FuncAnimation

def display_video(base64_images: list[str], interval=500):
    """
    Takes a list of Base64-encoded PNG images, decodes them, and displays them as an animated video using Matplotlib.
    
    Parameters:
    - base64_images: List of Base64-encoded strings of PNG images.
    - interval: Time delay between frames in milliseconds (default is 500ms).
    """
    
    # Function to convert a base64 string to a numpy array (image)
    def base64_to_image(base64_string):
        img_data = base64.b64decode(base64_string)
        img = Image.open(BytesIO(img_data))
        return np.array(img)

    # Convert all base64 strings to images (numpy arrays)
    images = [base64_to_image(img_str) for img_str in base64_images]

    # Set up the figure and axis for animation
    fig, ax = plt.subplots()

    # turn off the axis and any ticks
    ax.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])

    image_display = ax.imshow(images[0])  # Start with the first image

    # Update function for the animation
    def update(frame):
        image_display.set_array(images[frame])
        return [image_display]

    # Create the animation
    ani = FuncAnimation(
        fig, update, frames=len(images), interval=interval, blit=True
    )

    # Display the animation
    plt.show()











if __name__ == '__main__':
    main()