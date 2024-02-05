
import pdb
import xarray as xr
from matplotlib import pyplot as plt
from flowcast.pipeline import Variable


def solar_to_variable() -> Variable:
    data = xr.open_rasterio('DNI.tif')
    pdb.set_trace()


if __name__ == '__main__':
    solar_to_variable()
