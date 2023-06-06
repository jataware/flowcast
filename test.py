import xarray as xr
from matplotlib import pyplot as plt


import pdb


"""
[example question]
How many people will be exposed to extreme heat events (e.g., heatwaves) in the future?

Tasks:
- define what counts as an extreme heat event
- get projected population data
- combine the two and aggregate
- final plot should be a line graph of predicted people exposed (per year)
"""


def test():
    datapath = 'tasmax_Amon_CanESM5_ssp585_r13i1p2f1_gn_201501-210012.nc'
    data = xr.open_dataset(datapath, decode_coords='all')

    #plot the first time step
    data['tasmax'].isel(time=0).plot(); 
    plt.show()

    pdb.set_trace()
    ...



if __name__ == '__main__':
    test()